import torch
import triton
import triton.language as tl
import math
from typing import Optional

@triton.jit
def _flash_attention_forward_swa_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr, M_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    o_stride_b, o_stride_h, o_stride_s,
    m_stride_b, m_stride_h, m_stride_s,
    # Kernel parameters
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for the forward pass of causal FlashAttention with GQA, Sliding Window Attention, and Attention Sink.
    """
    # 1. Identify the block of queries and the batch/head to be processed.
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # GQA Logic: Map Query Head to Shared K/V Head
    q_per_kv = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // q_per_kv

    # 2. Initialize accumulators in SRAM.
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 3. Load the block of queries (Q_i).
    q_offsets = (q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M))
    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)
    
    qk_scale = softmax_scale
    
    # Process all off-diagonal blocks from 0 to the current query block
    for start_n in range(0, q_block_idx * BLOCK_M, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)

        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        # Compute attention scores
        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale
        
        # Apply the combined mask: (sliding | sink) & causal
        q_offsets_2d = q_offsets[:, None]
        k_offsets_2d = k_offsets[None, :]
        
        # Sliding window mask: key is within window_size of query
        sliding_mask = (k_offsets_2d <= q_offsets_2d) & (k_offsets_2d >= q_offsets_2d - (WINDOW_SIZE - 1))
        # Sink mask: key is a sink token (position < sink_size) and causal
        sink_mask = (k_offsets_2d < SINK_SIZE) & (k_offsets_2d <= q_offsets_2d)
        # Combined mask
        combined_mask = sliding_mask | sink_mask
        
        # Apply sequence length mask
        seq_mask = k_offsets_2d < SEQ_LEN
        combined_mask = combined_mask & seq_mask
        
        s_ij = tl.where(combined_mask, s_ij, -1e9)
        
        # Update online softmax statistics
        m_ij = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        exp_m_diff = tl.exp(m_i - m_new)
        acc = acc * exp_m_diff[:, None]
        l_i = l_i * exp_m_diff
        
        p_ij = tl.exp(s_ij - m_new[:, None])
        p_ij = tl.where(seq_mask, p_ij, 0.0)
        
        weighted_v = tl.dot(p_ij.to(tl.float32), v_block.to(tl.float32))
        acc = acc + weighted_v
        
        l_i = l_i + tl.sum(p_ij, axis=1)
        m_i = m_new

    # Diagonal Blocks
    diag_start = q_block_idx * BLOCK_M
    for start_n in range(diag_start, (q_block_idx + 1) * BLOCK_M, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_mask = k_offsets < SEQ_LEN

        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[:, None] * k_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        k_block = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)

        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)

        S = tl.sum(q_block[:, None, :] * k_block[None, :, :], 2) * qk_scale

        # Apply the combined mask: (sliding | sink) & causal
        q_offsets_2d = q_offsets[:, None]
        k_offsets_2d = k_offsets[None, :]
        
        # Sliding window mask: key is within window_size of query
        sliding_mask = (k_offsets_2d <= q_offsets_2d) & (k_offsets_2d >= q_offsets_2d - (WINDOW_SIZE - 1))
        # Sink mask: key is a sink token (position < sink_size) and causal
        sink_mask = (k_offsets_2d < SINK_SIZE) & (k_offsets_2d <= q_offsets_2d)
        # Combined mask
        combined_mask = sliding_mask | sink_mask
        
        # Apply sequence length mask
        seq_mask = (k_offsets[None, :] < SEQ_LEN) & (k_mask[None, :])
        combined_mask = combined_mask & seq_mask
        S = tl.where(combined_mask, S, -1e9)

        s_max = tl.max(S, 1)
        m_new = tl.maximum(m_i, s_max)

        p = tl.exp(S - m_new[:, None])

        exp_m_diff = tl.exp(m_i - m_new)
        l_new = l_i * exp_m_diff + tl.sum(p, 1)

        pv = tl.dot(p, v_block)
        acc = acc * exp_m_diff[:, None] + pv

        m_i = m_new
        l_i = l_new

    # 4. Normalize and write the final output block.
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]
    
    # Store L values (log-sum-exp) for backward pass
    # L = M + log(l)
    L_i = m_i + tl.log(l_i_safe)
    m_ptrs = M_ptr + batch_idx * m_stride_b + q_head_idx * m_stride_h + q_offsets * m_stride_s
    tl.store(m_ptrs, L_i, mask=q_offsets < SEQ_LEN)
    
    o_ptrs = O_ptr + batch_idx * o_stride_b + q_head_idx * o_stride_h + \
             (q_offsets[:, None] * o_stride_s + tl.arange(0, HEAD_DIM)[None, :])
             
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)

@triton.jit
def _flash_attention_backward_dq_swa_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr, DO_ptr, DQ_ptr, M_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    o_stride_b, o_stride_h, o_stride_s,
    do_stride_b, do_stride_h, do_stride_s,
    dq_stride_b, dq_stride_h, dq_stride_s,
    # Kernel parameters
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for computing dQ in the backward pass of SWA FlashAttention with sink tokens.
    """
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # GQA Logic: Map Query Head to Shared K/V Head
    q_per_kv = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // q_per_kv

    # Load query block
    q_offsets = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)

    # Load output block and gradient of output
    o_ptrs = O_ptr + batch_idx * o_stride_b + q_head_idx * o_stride_h + \
             (q_offsets[:, None] * o_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    o_block = tl.load(o_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)

    do_ptrs = DO_ptr + batch_idx * do_stride_b + q_head_idx * do_stride_h + \
              (q_offsets[:, None] * do_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    do_block = tl.load(do_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)

    # Load L values (log-sum-exp values from forward pass)
    L_ptrs = M_ptr + batch_idx * N_Q_HEADS * SEQ_LEN + q_head_idx * SEQ_LEN + q_offsets
    L_block = tl.load(L_ptrs, mask=q_offsets < SEQ_LEN, other=-float('inf'))

    # Compute delta = sum(dO * O)
    delta = tl.sum(do_block * o_block, axis=1)

    # Initialize dQ accumulator
    dq_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    qk_scale = softmax_scale

    # Process all key blocks that can attend to this query block (considering SWA + sink)
    max_k_block = (q_block_idx + 1) * BLOCK_M
    for start_n in range(0, max_k_block, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)

        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        # Compute attention scores
        s_ij = tl.dot(q_block, k_block) * qk_scale

        # Apply SWA + sink mask
        q_offsets_2d = q_offsets[:, None]
        k_offsets_2d = k_offsets[None, :]
        
        # Sliding window mask: key is within window_size of query
        sliding_mask = (k_offsets_2d <= q_offsets_2d) & (k_offsets_2d >= q_offsets_2d - (WINDOW_SIZE - 1))
        # Sink mask: key is a sink token (position < sink_size) and causal
        sink_mask = (k_offsets_2d < SINK_SIZE) & (k_offsets_2d <= q_offsets_2d)
        # Combined mask
        combined_mask = sliding_mask | sink_mask
        
        s_ij = tl.where(combined_mask, s_ij, -1e9)

        # Compute probabilities using the saved L values
        # P = exp(S - L)
        p_ij = tl.exp(s_ij - L_block[:, None])

        # Mask out invalid positions
        k_mask = k_offsets[None, :] < SEQ_LEN
        p_ij = tl.where(k_mask, p_ij, 0.0)

        # Compute dP = P * (dO @ V^T - delta)
        dov = tl.dot(do_block, tl.trans(v_block))
        dp = p_ij * (dov - delta[:, None])

        # Accumulate dQ += dP @ K^T
        # dp is [BLOCK_M, BLOCK_N], k_block is [HEAD_DIM, BLOCK_N]
        # We need dP @ K^T, so dp @ tl.trans(k_block)
        dq_acc += tl.dot(dp.to(tl.float32), tl.trans(k_block.to(tl.float32)))

    # Apply softmax scale
    dq_acc *= qk_scale

    # Store dQ
    dq_ptrs = DQ_ptr + batch_idx * dq_stride_b + q_head_idx * dq_stride_h + \
              (q_offsets[:, None] * dq_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    tl.store(dq_ptrs, dq_acc.to(DQ_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)


@triton.jit
def _flash_attention_backward_dkv_swa_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr, DO_ptr, DK_ptr, DV_ptr, M_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    o_stride_b, o_stride_h, o_stride_s,
    do_stride_b, do_stride_h, do_stride_s,
    dk_stride_b, dk_stride_h, dk_stride_s,
    dv_stride_b, dv_stride_h, dv_stride_s,
    # Kernel parameters
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for computing dK and dV in the backward pass of SWA FlashAttention with sink tokens.
    """
    k_block_idx = tl.program_id(axis=0)
    batch_kv_head_idx = tl.program_id(axis=1)
    
    batch_idx = batch_kv_head_idx // N_KV_HEADS
    kv_head_idx = batch_kv_head_idx % N_KV_HEADS

    # Load key and value blocks
    k_offsets = k_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
             (k_offsets[:, None] * k_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    k_block = tl.load(k_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

    v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
             (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

    # Initialize dK and dV accumulators
    dk_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    
    qk_scale = softmax_scale

    # GQA: Process all query heads that share this K/V head
    q_per_kv = N_Q_HEADS // N_KV_HEADS
    
    for q_group_offset in range(q_per_kv):
        q_head_idx = kv_head_idx * q_per_kv + q_group_offset
        
        # Process query blocks that can attend to this key block with SWA + sink logic
        # For SWA: query at position q can attend to key at position k if:
        # 1. k <= q (causal)
        # 2. k >= q - (window_size - 1) (sliding window) OR k < sink_size (sink)
        
        # Simplified approach: process all query blocks that could potentially attend to this key block
        # We'll let the mask handle the actual filtering
        min_q_block = 0
        max_q_block = tl.cdiv(SEQ_LEN, BLOCK_M)
        
        for q_block_idx in range(min_q_block, max_q_block):
            q_offsets = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
            
            # Load query block
            q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
                     (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
            q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)

            # Load output and gradient of output
            o_ptrs = O_ptr + batch_idx * o_stride_b + q_head_idx * o_stride_h + \
                     (q_offsets[:, None] * o_stride_s + tl.arange(0, HEAD_DIM)[None, :])
            o_block = tl.load(o_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)

            do_ptrs = DO_ptr + batch_idx * do_stride_b + q_head_idx * do_stride_h + \
                      (q_offsets[:, None] * do_stride_s + tl.arange(0, HEAD_DIM)[None, :])
            do_block = tl.load(do_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)

            # Load L values (log-sum-exp values from forward pass)
            L_ptrs = M_ptr + batch_idx * N_Q_HEADS * SEQ_LEN + q_head_idx * SEQ_LEN + q_offsets
            L_block = tl.load(L_ptrs, mask=q_offsets < SEQ_LEN, other=-float('inf'))

            # Compute delta = sum(dO * O)
            delta = tl.sum(do_block * o_block, axis=1)

            # Compute attention scores
            s_ij = tl.dot(q_block, tl.trans(k_block)) * qk_scale

            # Apply SWA + sink mask
            q_offsets_2d = q_offsets[:, None]
            k_offsets_2d = k_offsets[None, :]
            
            # Sliding window mask: key is within window_size of query
            sliding_mask = (k_offsets_2d <= q_offsets_2d) & (k_offsets_2d >= q_offsets_2d - (WINDOW_SIZE - 1))
            # Sink mask: key is a sink token (position < sink_size) and causal
            sink_mask = (k_offsets_2d < SINK_SIZE) & (k_offsets_2d <= q_offsets_2d)
            # Combined mask
            combined_mask = sliding_mask | sink_mask

            s_ij = tl.where(combined_mask, s_ij, -1e9)

            # Compute probabilities using the saved L values
            # P = exp(S - L)
            p_ij = tl.exp(s_ij - L_block[:, None])

            # Mask out invalid positions
            k_mask = k_offsets[None, :] < SEQ_LEN
            q_mask = q_offsets[:, None] < SEQ_LEN
            combined_valid_mask = k_mask & q_mask
            p_ij = tl.where(combined_valid_mask, p_ij, 0.0)

            # Compute dP = P * (dO @ V^T - delta)
            dov = tl.dot(do_block, tl.trans(v_block))
            dp = p_ij * (dov - delta[:, None])

            # Accumulate dK += dP^T @ Q
            # dp is [BLOCK_M, BLOCK_N], q_block is [BLOCK_M, HEAD_DIM]
            # We need dP^T @ Q to get [BLOCK_N, HEAD_DIM]
            dk_acc += tl.dot(tl.trans(dp.to(tl.float32)), q_block.to(tl.float32))

            # Accumulate dV += P^T @ dO
            # p_ij is [BLOCK_M, BLOCK_N], do_block is [BLOCK_M, HEAD_DIM]
            # We need P^T @ dO, so tl.trans(p_ij) @ do_block
            dv_acc += tl.dot(tl.trans(p_ij.to(tl.float32)), do_block.to(tl.float32))

    # Apply softmax scale to dK
    dk_acc *= qk_scale

    # Store dK and dV
    dk_ptrs = DK_ptr + batch_idx * dk_stride_b + kv_head_idx * dk_stride_h + \
              (k_offsets[:, None] * dk_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    tl.store(dk_ptrs, dk_acc.to(DK_ptr.dtype.element_ty), mask=k_offsets[:, None] < SEQ_LEN)

    dv_ptrs = DV_ptr + batch_idx * dv_stride_b + kv_head_idx * dv_stride_h + \
              (k_offsets[:, None] * dv_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    tl.store(dv_ptrs, dv_acc.to(DV_ptr.dtype.element_ty), mask=k_offsets[:, None] < SEQ_LEN)

class FlashSWDAWithSink(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, window_size, sink_size, is_causal=True, softmax_scale=None):
        assert is_causal, "Currently, only causal attention is supported"

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])

        batch, n_q_heads, seq_len, head_dim = q.shape
        _, n_kv_heads, _, _ = k.shape

        assert q.shape[0] == v.shape[0] and q.shape[2] == v.shape[2] and q.shape[3] == v.shape[3], "Query and Value shapes must be compatible except for num_heads"
        assert k.shape[0] == v.shape[0] and k.shape[1] == v.shape[1] and k.shape[2] == v.shape[2] and k.shape[3] == v.shape[3], "Key and Value shapes must be the same"
        assert head_dim <= 128, "Head dimension must be less than or equal to 128"
        assert n_q_heads % n_kv_heads == 0, "Number of query heads must be divisible by number of K/V heads"

        o = torch.empty_like(q)
        M = torch.empty((batch, n_q_heads, seq_len), device=q.device, dtype=torch.float32)

        BLOCK_M, BLOCK_N = 128, 64
        grid = (math.ceil(seq_len / BLOCK_M), batch * n_q_heads)

        _flash_attention_forward_swa_kernel[grid](
            q, k, v, o, M,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            M.stride(0), M.stride(1), M.stride(2),
            softmax_scale,
            seq_len,
            n_q_heads,
            n_kv_heads,
            WINDOW_SIZE=window_size,
            SINK_SIZE=sink_size,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.softmax_scale = softmax_scale
        ctx.window_size = window_size
        ctx.sink_size = sink_size
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        softmax_scale = ctx.softmax_scale
        window_size = ctx.window_size
        sink_size = ctx.sink_size

        batch, n_q_heads, seq_len, head_dim = q.shape
        n_kv_heads = k.shape[1]

        dq = torch.empty_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        
        BLOCK_M, BLOCK_N = 128, 64
        
        # Compute dQ
        grid_dq = (math.ceil(seq_len / BLOCK_M), batch * n_q_heads)
        _flash_attention_backward_dq_swa_kernel[grid_dq](
            q, k, v, o, do, dq, M,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2),
            softmax_scale,
            seq_len,
            n_q_heads,
            n_kv_heads,
            WINDOW_SIZE=window_size,
            SINK_SIZE=sink_size,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        
        # Compute dK and dV
        grid_dkv = (math.ceil(seq_len / BLOCK_N), batch * n_kv_heads)
        _flash_attention_backward_dkv_swa_kernel[grid_dkv](
            q, k, v, o, do, dk, dv, M,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            dk.stride(0), dk.stride(1), dk.stride(2),
            dv.stride(0), dv.stride(1), dv.stride(2),
            softmax_scale,
            seq_len,
            n_q_heads,
            n_kv_heads,
            WINDOW_SIZE=window_size,
            SINK_SIZE=sink_size,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None, None
    
def flash_swda_with_sink(q, k, v, window_size: int, sink_size: int = 0, is_causal: bool = True, scale: Optional[float] = None):
    return FlashSWDAWithSink.apply(q, k, v, window_size, sink_size, is_causal, scale)
# problem_8.py
import torch
import triton
import triton.language as tl
import math
from typing import Optional

@triton.jit
def _flash_attention_forward_gqa_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr, M_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    # Kernel parameters
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for the forward pass of causal FlashAttention with GQA.
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
    
    # Phase 1: Off-Diagonal Blocks
    for start_n in range(0, q_block_idx * BLOCK_M, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)

        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        # Compute the attention scores (S_ij).
        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale
        
        # Update the online softmax statistics (m_i, l_i) and the accumulator (acc).
        # row-wise max in this tile
        m_ij = tl.max(s_ij, axis=1)

        # new running max
        m_new = tl.maximum(m_i, m_ij)

        # rescale previous accumulator/denominator
        exp_m_diff = tl.exp(m_i - m_new)   # [BLOCK_M]
        acc = acc * exp_m_diff[:, None]     # [BLOCK_M, HEAD_DIM]
        l_i = l_i * exp_m_diff              # [BLOCK_M]

        # compute probabilities
        p_ij = tl.exp(s_ij - m_new[:, None])  # [BLOCK_M, BLOCK_N]

        # mask
        k_mask = (k_offsets[None, :] < SEQ_LEN)  # [1, BLOCK_N]
        p_ij = tl.where(k_mask, p_ij, 0.0)

        # weighted update
        weighted_v = tl.dot(p_ij.to(tl.float32), v_block.to(tl.float32))
        acc = acc + weighted_v

        # denominator update
        l_i = l_i + tl.sum(p_ij, axis=1)

        # update running max
        m_i = m_new

    # Phase 2: Diagonal Blocks
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

        # Causal mask: allow k_pos <= q_pos
        causal_mask = (k_offsets[None, :] <= q_offsets[:, None]) & (k_mask[None, :])
        S = tl.where(causal_mask, S, -1e9)

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
    m_ptrs = M_ptr + batch_idx * N_Q_HEADS * SEQ_LEN + q_head_idx * SEQ_LEN + q_offsets
    tl.store(m_ptrs, L_i, mask=q_offsets < SEQ_LEN)
    
    o_ptrs = O_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
             
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)


@triton.jit
def _flash_attention_backward_dq_kernel(
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
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for computing dQ in the backward pass of GQA FlashAttention.
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

    # Process all key blocks that can attend to this query block (causal)
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

        # Apply causal mask
        causal_mask = k_offsets[None, :] <= q_offsets[:, None]
        s_ij = tl.where(causal_mask, s_ij, -1e9)

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
def _flash_attention_backward_dkv_kernel(
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
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for computing dK and dV in the backward pass of GQA FlashAttention.
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
        
        # Process all query blocks that can attend to this key block (causal)
        # For causal attention, query at position q can attend to key at position k if q >= k
        # So for key block starting at k_block_idx * BLOCK_N, 
        # minimum query block is the one containing position k_block_idx * BLOCK_N
        min_q_block = (k_block_idx * BLOCK_N) // BLOCK_M
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

            # Apply causal mask
            causal_mask = k_offsets[None, :] <= q_offsets[:, None]
            s_ij = tl.where(causal_mask, s_ij, -1e9)

            # Compute probabilities using the saved L values
            # P = exp(S - L)
            p_ij = tl.exp(s_ij - L_block[:, None])

            # Mask out invalid positions
            k_mask = k_offsets[None, :] < SEQ_LEN
            q_mask = q_offsets[:, None] < SEQ_LEN
            combined_mask = k_mask & q_mask
            p_ij = tl.where(combined_mask, p_ij, 0.0)

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


class FlashAttention2Function(torch.autograd.Function):
    """
    Triton implementation of FlashAttention-2, supports causal attention and GQA.
    """
    @staticmethod
    def forward(ctx, q, k, v, is_causal=True, softmax_scale: Optional[float] = None):
        batch, n_heads, seq_len, head_dim = q.shape
        n_kv_heads = k.shape[1]

        assert is_causal, "This kernel only supports causal attention"
        assert n_heads % n_kv_heads == 0, "num_attention_heads must be divisible by num_kv_heads"

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        o = torch.empty_like(q)
        M = torch.empty((batch, n_heads, seq_len), device=q.device, dtype=torch.float32)

        BLOCK_M, BLOCK_N = 128, 64
        grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)
        
        # Forward kernel implementation
        _flash_attention_forward_gqa_kernel[grid](
            q, k, v, o, M,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            softmax_scale,
            seq_len,
            n_heads,
            n_kv_heads,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.softmax_scale = softmax_scale
        ctx.num_heads = n_heads
        ctx.num_kv_heads = n_kv_heads
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        batch, n_heads, seq_len, head_dim = q.shape
        n_kv_heads = ctx.num_kv_heads

        dq = torch.empty_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        # [OPTIONAL BONUS] STUDENT IMPLEMENTATION REQUIRED
        # Implement the Triton backward kernel for GQA from scratch.
        # You should:
        #   1. Precompute delta = sum(dO * O)
        #   2. Recompute attention probabilities P = softmax(QK^T)
        #   3. Use delta + dO to accumulate gradients for dq, dk, dv
        #   4. Respect GQA mapping and causal mask
        
        BLOCK_M, BLOCK_N = 128, 64
        
        # Compute dQ
        grid_dq = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)
        _flash_attention_backward_dq_kernel[grid_dq](
            q, k, v, o, do, dq, M,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2),
            ctx.softmax_scale,
            seq_len,
            n_heads,
            n_kv_heads,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        
        # Compute dK and dV
        grid_dkv = (triton.cdiv(seq_len, BLOCK_N), batch * n_kv_heads)
        _flash_attention_backward_dkv_kernel[grid_dkv](
            q, k, v, o, do, dk, dv, M,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            dk.stride(0), dk.stride(1), dk.stride(2),
            dv.stride(0), dv.stride(1), dv.stride(2),
            ctx.softmax_scale,
            seq_len,
            n_heads,
            n_kv_heads,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        
        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None


def flash_attention_gqa(q, k, v, is_causal=True, softmax_scale=None):
    return FlashAttention2Function.apply(q, k, v, is_causal, softmax_scale)
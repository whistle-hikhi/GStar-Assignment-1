import torch
import triton
import triton.language as tl
import math

@triton.jit
def _flash_attention_forward_swa_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    # Kernel parameters
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    WINDOW_SIZE: tl.constexpr,
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel template for Sliding Window Attention (SWA) with GQA.
    """
    # 1. Boilerplate setup
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # --- STUDENT IMPLEMENTATION REQUIRED (Part 1: GQA Logic) ---
    # This problem combines GQA and SWA. First, implement the GQA logic.
    # 1. Calculate the number of query heads per group.
    # 2. Determine the correct kv_head_idx for the current q_head_idx.
    
    q_per_kv = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // q_per_kv 
    # --- END OF GQA IMPLEMENTATION ---


    # 2. Initialize accumulators
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 3. Load query block
    q_offsets = (q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M))
    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)
    
    qk_scale = softmax_scale * 1.44269504

    # --- STUDENT IMPLEMENTATION REQUIRED (Part 2: SWA Logic) ---
    # Now, implement the "sliding window" by changing the loop bounds.
    # The kernel should only attend to the `WINDOW_SIZE` most recent key/value tokens.
    # 1. Calculate the starting position of the attention window (window_start).
    # 2. Modify the range of the Phase 1 loop to start from your window_start.

    window_start = max(0, (q_block_idx * BLOCK_M) - WINDOW_SIZE)
    # --- Phase 1: Off-Diagonal Blocks (within the window) ---
    for start_n in range(window_start, q_block_idx * BLOCK_M, BLOCK_N):
        # STUDENT IMPLEMENTATION REQUIRED (Part 3: SWA Logic)
        # Hint: You might need to apply the per-element sliding window mask to s_ij.
        #    - A score is invalid if `(query_offset - key_offset) >= WINDOW_SIZE`.
        
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
        
        # Apply sliding window mask: invalid if (query_offset - key_offset) >= WINDOW_SIZE
        q_offsets_2d = q_offsets[:, None]
        k_offsets_2d = k_offsets[None, :]
        window_mask = (q_offsets_2d - k_offsets_2d) < WINDOW_SIZE
        s_ij = tl.where(window_mask, s_ij, -1e9)
        
        # Apply causal mask (query position >= key position)
        causal_mask = q_offsets_2d >= k_offsets_2d
        s_ij = tl.where(causal_mask, s_ij, -1e9)
        
        # Update the online softmax statistics (m_i, l_i) and the accumulator (acc).
        # row-wise max in this tile
        m_ij = tl.max(s_ij, axis=1)

        # new running max
        m_new = tl.maximum(m_i, m_ij)

        # rescale previous accumulator/denominator
        exp_m_diff = tl.exp2(m_i - m_new)   # [BLOCK_M]
        acc = acc * exp_m_diff[:, None]     # [BLOCK_M, HEAD_DIM]
        l_i = l_i * exp_m_diff              # [BLOCK_M]

        # compute probabilities
        p_ij = tl.exp2(s_ij - m_new[:, None])  # [BLOCK_M, BLOCK_N]

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

    # --- Phase 2: Diagonal Blocks ---
    diag_start = q_block_idx * BLOCK_M
    for start_n in range(diag_start, (q_block_idx + 1) * BLOCK_M, BLOCK_N):
        # STUDENT IMPLEMENTATION REQUIRED
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_mask = k_offsets < SEQ_LEN

        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[:, None] * k_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        k_block = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)

        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)

        S = tl.sum(q_block[:, None, :] * k_block[None, :, :], 2) * qk_scale

        # Apply sliding window mask: invalid if (query_offset - key_offset) >= WINDOW_SIZE
        q_offsets_2d = q_offsets[:, None]
        k_offsets_2d = k_offsets[None, :]
        window_mask = (q_offsets_2d - k_offsets_2d) < WINDOW_SIZE
        
        # Causal mask: allow k_pos <= q_pos
        causal_mask = (k_offsets[None, :] <= q_offsets[:, None]) & (k_mask[None, :])
        
        # Combined mask: both causal and sliding window
        combined_mask = causal_mask & window_mask
        S = tl.where(combined_mask, S, -1e9)

        s_max = tl.max(S, 1)
        m_new = tl.maximum(m_i, s_max)

        p = tl.exp2(S - m_new[:, None])

        exp_m_diff = tl.exp2(m_i - m_new)
        l_new = l_i * exp_m_diff + tl.sum(p, 1)

        pv = tl.dot(p, v_block)
        acc = acc * exp_m_diff[:, None] + pv

        m_i = m_new
        l_i = l_new
    # --- END OF SWA IMPLEMENTATION ---


    # 4. Normalize and write the final output block.
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]
    
    o_ptrs = O_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
             
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)


def flash_attention_forward(q, k, v, is_causal=True, window_size=128):
    
    """
    Python wrapper for the SWA-enabled GQA causal FlashAttention kernel.
    """
    batch, n_q_heads, seq_len, head_dim = q.shape
    n_kv_heads = k.shape[1]
    
    assert n_q_heads % n_kv_heads == 0, "Number of query heads must be divisible by number of K/V heads"
    assert is_causal, "This kernel is only supported for causal attention"

    o = torch.empty_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_q_heads)

    # if window_size != 4096:
    #     raise ValueError("This kernel is compiled for a fixed window size of 4096")

    _flash_attention_forward_swa_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        softmax_scale,
        seq_len,
        n_q_heads,
        n_kv_heads,
        WINDOW_SIZE=window_size,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return o
import torch
import triton
import triton.language as tl
import math

@triton.jit
def _flash_attention_forward_gqa_kernel(
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
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel template for the forward pass of causal FlashAttention with GQA.
    """
    # 1. Identify the block of queries and the batch/head to be processed.
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # --- STUDENT IMPLEMENTATION REQUIRED HERE (Part 1) ---
    # Your goal is to map the current query head (q_head_idx) to its corresponding shared key/value head (kv_head_idx).
    # 1. Calculate how many query heads are in each group.
    # 2. Use integer division to find the correct kv_head_idx.
    # --- STUDENT IMPLEMENTATION REQUIRED HERE (Part 1) ---
    # Map query head -> KV head using contiguous grouping
    q_per_kv = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // q_per_kv
    # --- END OF STUDENT IMPLEMENTATION ---


    # 2. Initialize accumulators in SRAM.
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 3. Load the block of queries (Q_i).
    q_offsets = (q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M))
    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)
    
    qk_scale = softmax_scale * 1.44269504
    
    # --- Phase 1: Off-Diagonal Blocks ---
    for start_n in range(0, q_block_idx * BLOCK_M, BLOCK_N):
        # --- STUDENT IMPLEMENTATION REQUIRED HERE (Part 2) ---
        k_offsets = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[:, None] * k_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])

        k_block = tl.load(k_ptrs, mask=(k_offsets[:, None] < SEQ_LEN), other=0.0)
        v_block = tl.load(v_ptrs, mask=(k_offsets[:, None] < SEQ_LEN), other=0.0)

        # compute scores and mask out invalid key positions
        # cancel ln(2) factor locally if present in qk_scale
        inv_ln2 = 1.0 / 1.44269504
        scores = tl.dot(q_block, k_block.T) * qk_scale * inv_ln2  # [BLOCK_M, BLOCK_N]

        valid_k = k_offsets < SEQ_LEN                     # [BLOCK_N]
        neg_inf = -1e9
        scores = tl.where(valid_k[None, :], scores, neg_inf)

        # numerically-stable online softmax merge
        s_max = tl.max(scores, axis=1)          # [BLOCK_M]
        m_j = tl.maximum(m_i, s_max)            # [BLOCK_M]

        exp_mi_mj = tl.exp(m_i - m_j)                     # [BLOCK_M]
        exp_scores = tl.exp(scores - m_j[:, None])        # [BLOCK_M, BLOCK_N]
        l_new = exp_mi_mj * l_i + tl.sum(exp_scores, axis=1)

        # exp_scores @ V_block via broadcast+sum
        tmp = tl.sum(exp_scores[:, :, None] * v_block[None, :, :], axis=1)
        acc = exp_mi_mj[:, None] * acc + tmp

        m_i = m_j
        l_i = l_new
        # --- END OF STUDENT IMPLEMENTATION ---

    # --- Phase 2: Diagonal Blocks ---
    diag_start = q_block_idx * BLOCK_M
    for start_n in range(diag_start, (q_block_idx + 1) * BLOCK_M, BLOCK_N):
        # --- STUDENT IMPLEMENTATION REQUIRED HERE (Part 3) ---
        # 1. Modify the pointer arithmetic for K and V to use your `kv_head_idx`.
        # 2. Reuse your working implementation for the masked online softmax
        #    update from your solution to Problem 4.
        # --- STUDENT IMPLEMENTATION REQUIRED HERE (Part 3) ---
        k_offsets = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[:, None] * k_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])

        k_block = tl.load(k_ptrs, mask=(k_offsets[:, None] < SEQ_LEN), other=0.0)
        v_block = tl.load(v_ptrs, mask=(k_offsets[:, None] < SEQ_LEN), other=0.0)

        # compute scores (cancel ln(2) factor locally)
        inv_ln2 = 1.0 / 1.44269504
        scores = tl.dot(q_block, k_block.T) * qk_scale * inv_ln2  # [BLOCK_M, BLOCK_N]

        # causal mask: allow k_pos <= q_pos
        key_pos_row = k_offsets[None, :]   # [1, BLOCK_N]
        q_pos_col = q_offsets[:, None]     # [BLOCK_M, 1]
        valid_k = key_pos_row < SEQ_LEN
        causal_mask = valid_k & (key_pos_row <= q_pos_col)

        neg_inf = -1e9
        scores = tl.where(causal_mask, scores, neg_inf)

        s_max = tl.max(scores, axis=1)
        m_j = tl.maximum(m_i, s_max)

        exp_mi_mj = tl.exp(m_i - m_j)
        exp_scores = tl.exp(scores - m_j[:, None])
        l_new = exp_mi_mj * l_i + tl.sum(exp_scores, axis=1)

        tmp = tl.sum(exp_scores[:, :, None] * v_block[None, :, :], axis=1)
        acc = exp_mi_mj[:, None] * acc + tmp

        m_i = m_j
        l_i = l_new
        # --- END OF STUDENT IMPLEMENTATION ---

    # 4. Normalize and write the final output block.
    l_i_safe = l_i[:, None] + 1e-6
    acc = acc / l_i_safe
    
    o_ptrs = O_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
             
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)


def flash_attention_forward(q, k, v, is_causal=True):
    """
    Python wrapper for the GQA-enabled causal FlashAttention kernel.
    """
    batch, n_q_heads, seq_len, head_dim = q.shape
    n_kv_heads = k.shape[1]
    
    assert n_q_heads % n_kv_heads == 0, "Number of query heads must be divisible by number of K/V heads"
    
    o = torch.empty_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_q_heads)

    _flash_attention_forward_gqa_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        softmax_scale,
        seq_len,
        n_q_heads,
        n_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return o
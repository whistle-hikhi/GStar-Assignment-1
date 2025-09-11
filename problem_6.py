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
    
    # Number of query heads served by each kv head (integer division; assert divides evenly in python wrapper)
    q_heads_per_kv = N_Q_HEADS // N_KV_HEADS
    # Map the q_head_idx to its corresponding kv head index
    kv_head_idx = q_head_idx // q_heads_per_kv
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

    # For causal attention: each query at position i attends to keys in [max(0, i-WINDOW_SIZE+1), i]
    # For the block at q_block_idx, the highest query position in the block is (q_block_idx*BLOCK_M + BLOCK_M - 1)
    # We compute the earliest key position any query in this block could attend to, then align it down to BLOCK_N
    max_query_pos = q_block_idx * BLOCK_M + (BLOCK_M - 1)
    earliest_key_pos = max_query_pos - (WINDOW_SIZE - 1)
    if earliest_key_pos < 0:
        earliest_key_pos = 0
    # Align to BLOCK_N boundary (start of a key-block)
    window_start = (earliest_key_pos // BLOCK_N) * BLOCK_N
    # --- Phase 1: Off-Diagonal Blocks (within the window) ---
    for start_n in range(window_start, q_block_idx * BLOCK_M, BLOCK_N):
        # STUDENT IMPLEMENTATION REQUIRED (Part 3: SWA Logic)
        # Hint: You might need to apply the per-element sliding window mask to s_ij.
        #    - A score is invalid if `(query_offset - key_offset) >= WINDOW_SIZE`.
        # Build key offsets for this key block
        k_offsets = (start_n + tl.arange(0, BLOCK_N))
        # Ptrs for K and V using the kv_head_idx computed above
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[:, None] * k_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        k_block = tl.load(k_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        # Compute raw scores: (BLOCK_M x HEAD_DIM) @ (HEAD_DIM x BLOCK_N) -> (BLOCK_M x BLOCK_N)
        s = tl.dot(q_block, k_block, trans_b=True) * qk_scale  # shape [BLOCK_M, BLOCK_N]

        # Build mask for valid positions:
        # causal: key_pos <= query_pos  -> (q_offset - k_offset) >= 0
        # sliding window: (q_offset - k_offset) < WINDOW_SIZE
        # also ensure key_pos < SEQ_LEN and query_pos < SEQ_LEN (query mask done on load)
        rel = (q_offsets[:, None] - k_offsets[None, :])  # shape [BLOCK_M, BLOCK_N]
        valid_mask = (rel >= 0) & (rel < WINDOW_SIZE) & (k_offsets[None, :] < SEQ_LEN)

        # For invalid positions set score to -inf to ignore them in softmax
        s = tl.where(valid_mask, s, -float('inf'))

        # Numerically stable softmax accumulation (online)
        s_max = tl.max(s, axis=1)  # shape [BLOCK_M]
        m_new = tl.maximum(m_i, s_max)
        exp_s = tl.exp(s - m_new[:, None])
        l_new = tl.exp(m_i - m_new) * l_i + tl.sum(exp_s, axis=1)
        # update acc: acc * exp(m_i - m_new) + exp_s @ v_block
        # compute exp_s @ v_block (shape [BLOCK_M, HEAD_DIM])
        weighted_vals = tl.dot(exp_s, v_block)  # uses exp_s (BLOCK_M x BLOCK_N) & v_block (BLOCK_N x HEAD_DIM)
        acc = acc * tl.exp(m_i - m_new)[:, None] + weighted_vals

        # commit updated m_i and l_i
        m_i = m_new
        l_i = l_new

    # --- Phase 2: Diagonal Blocks ---
    diag_start = q_block_idx * BLOCK_M
    for start_n in range(diag_start, (q_block_idx + 1) * BLOCK_M, BLOCK_N):
        # Diagonal blocks: keys and queries from roughly the same block.
        k_offsets = (start_n + tl.arange(0, BLOCK_N))
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[:, None] * k_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        k_block = tl.load(k_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        s = tl.dot(q_block, k_block, trans_b=True) * qk_scale  # [BLOCK_M, BLOCK_N]

        # For diagonal, enforce causal per-element: key_pos <= query_pos
        rel = (q_offsets[:, None] - k_offsets[None, :])
        valid_mask = (rel >= 0) & (rel < WINDOW_SIZE) & (k_offsets[None, :] < SEQ_LEN)

        s = tl.where(valid_mask, s, -float('inf'))

        # Numerically stable softmax accumulation (online) same as above
        s_max = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, s_max)
        exp_s = tl.exp(s - m_new[:, None])
        l_new = tl.exp(m_i - m_new) * l_i + tl.sum(exp_s, axis=1)
        weighted_vals = tl.dot(exp_s, v_block)
        acc = acc * tl.exp(m_i - m_new)[:, None] + weighted_vals
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

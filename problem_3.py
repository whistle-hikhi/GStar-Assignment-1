import torch
import triton
import triton.language as tl
import math

@triton.jit
def _flash_attention_forward_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    # Kernel parameters
    softmax_scale,
    SEQ_LEN,
    N_HEADS,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for the forward pass of FlashAttention-2 (non-causal).
    """
    # 1. Identify the block of queries and the batch/head to be processed.
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    
    batch_idx = batch_head_idx // N_HEADS
    head_idx = batch_head_idx % N_HEADS

    # 2. Initialize pointers and accumulators for the online softmax.
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 3. Load the block of queries (Q_i).
    q_offsets = (q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M))
    q_ptrs = Q_ptr + batch_idx * q_stride_b + head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)
    
    # PyTorch softmax is exp(x), Triton is exp2(x * log2(e)), log2(e) is approx 1.44269504
    qk_scale = softmax_scale * 1.44269504

    # 4. Main loop: Iterate over blocks of keys (K_j) and values (V_j).
    for start_n in range(0, SEQ_LEN, BLOCK_N):
        # - Load K_j
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        
        # Compute attention scores S_ij = Q_i * K_j^T
        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale

        # Load V_j
        v_ptrs = V_ptr + batch_idx * v_stride_b + head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        # ---------------- ONLINE SOFTMAX UPDATE ----------------
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
        # --------------------------------------------------------

    # 5. Normalize the accumulator and write the output block.
    l_i_safe = l_i[:, None] + 1e-6
    acc = acc / l_i_safe
    
    o_ptrs = O_ptr + batch_idx * q_stride_b + head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)

def flash_attention_forward(q, k, v, is_causal=False):
    """
    Minimal Python wrapper for the FlashAttention-2 forward pass.
    """
    batch, n_heads, seq_len, head_dim = q.shape
    o = torch.empty_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)

    _flash_attention_forward_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        softmax_scale,
        seq_len,
        n_heads,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return o
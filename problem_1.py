import torch
import torch.nn as nn
import math

class FlashAttention2Function(torch.autograd.Function):
    """
    A pure PyTorch implementation of the FlashAttention-2 forward pass.
    This version is a template for student implementation.
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # Get dimensions from input tensors following the (B, H, N, D) convention
        B, H, N_Q, D_H = Q.shape
        _, _, N_K, _ = K.shape

        # Define tile sizes
        Q_TILE_SIZE = 128
        K_TILE_SIZE = 128
        
        N_Q_tiles = math.ceil(N_Q / Q_TILE_SIZE)
        N_K_tiles = math.ceil(N_K / K_TILE_SIZE)

        # Initialize final output tensors
        O_final = torch.zeros_like(Q, dtype=Q.dtype)
        L_final = torch.zeros((B, H, N_Q), device=Q.device, dtype=torch.float32)
        
        scale = 1.0 / math.sqrt(D_H)

        # Main loops: Iterate over each batch and head
        for b in range(B):
            for h in range(H):
                Q_bh = Q[b, h, :, :]
                K_bh = K[b, h, :, :]
                V_bh = V[b, h, :, :]

                # Loop over query tiles
                for i in range(N_Q_tiles):
                    q_start = i * Q_TILE_SIZE
                    q_end = min((i + 1) * Q_TILE_SIZE, N_Q)
                    Q_tile = Q_bh[q_start:q_end, :]

                    q_len = q_end - q_start

                    # Initialize accumulators for this query tile (use float32 for numerical stability)
                    o_i = torch.zeros((q_len, D_H), device=Q.device, dtype=torch.float32)
                    l_i = torch.zeros((q_len,), device=Q.device, dtype=torch.float32)
                    m_i = torch.full((q_len,), -float('inf'), device=Q.device, dtype=torch.float32)

                    # Inner loop over key/value tiles
                    for j in range(N_K_tiles):
                        k_start = j * K_TILE_SIZE
                        k_end = min((j + 1) * K_TILE_SIZE, N_K)

                        K_tile = K_bh[k_start:k_end, :]
                        V_tile = V_bh[k_start:k_end, :]
                        
                        # Compute attention scores for this tile (cast to float32 for stable math)
                        S_ij = (Q_tile @ K_tile.transpose(-1, -2)) * scale
                        S_ij = S_ij.to(torch.float32)

                        # 1. Apply causal masking if is_causal is True.
                        if is_causal:
                            # absolute indices of queries and keys
                            q_idx = torch.arange(q_start, q_end, device=Q.device, dtype=torch.long)
                            k_idx = torch.arange(k_start, k_end, device=Q.device, dtype=torch.long)
                            # mask positions where key index > query index
                            # (q_len, k_len) boolean
                            causal_mask = (k_idx.unsqueeze(0) > q_idx.unsqueeze(1))
                            # set masked scores to -inf so exp -> 0
                            S_ij.masked_fill_(causal_mask, -float('inf'))

                        # 2. Compute the new running maximum m_new per query position
                        # max over key dim for this tile
                        # S_ij.max(dim=1).values has shape (q_len,)
                        s_ij_max = torch.where(
                            torch.isfinite(S_ij).any(dim=1),
                            S_ij.max(dim=1).values,
                            torch.full((q_len,), -float('inf'), device=Q.device, dtype=torch.float32)
                        )
                        m_new = torch.maximum(m_i, s_ij_max)

                        # 3. Rescale the previous accumulators (o_i, l_i)
                        # factor = exp(m_i - m_new) (shape q_len)
                        # where m_new may equal -inf; handle numerically
                        neg_diff = (m_i - m_new)
                        # exp might underflow but in float32 it's okay for typical scales
                        rescale = torch.exp(neg_diff)
                        # broadcast to match o_i shape for multiplication
                        o_i = o_i * rescale.unsqueeze(-1)
                        l_i = l_i * rescale

                        # 4. Compute the probabilities for the current tile:
                        # P_tilde_ij = exp(S_ij - m_new.unsqueeze(-1))
                        # Note: if S_ij had -inf entries from masking, exp -> 0
                        P_tilde = torch.exp(S_ij - m_new.unsqueeze(-1))

                        # 5. Accumulate the current tile's contribution to the accumulators
                        # l_i += sum_k P_tilde
                        l_i = l_i + P_tilde.sum(dim=1)

                        # o_i += P_tilde @ V_tile (ensure V_tile is float32 for stable accumulation)
                        o_i = o_i + (P_tilde @ V_tile.to(torch.float32))

                        # 6. Update the running max for the next iteration
                        m_i = m_new

                    # After iterating through all key tiles, normalize the output
                    # This part is provided for you. It handles the final division safely.
                    l_i_reciprocal = torch.where(l_i > 0, 1.0 / l_i, 0)
                    o_i_normalized = o_i * l_i_reciprocal.unsqueeze(-1)
                    
                    L_tile = m_i + torch.log(l_i)

                    # Write results for this tile back to the final output tensors
                    # Cast o_i_normalized back to input dtype
                    O_final[b, h, q_start:q_end, :] = o_i_normalized.to(Q.dtype)
                    L_final[b, h, q_start:q_end] = L_tile
        
        O_final = O_final.to(Q.dtype)

        ctx.save_for_backward(Q, K, V, O_final, L_final)
        ctx.is_causal = is_causal
 
        return O_final, L_final
    
    @staticmethod
    def backward(ctx, grad_out, grad_L):
        raise NotImplementedError("Backward pass not yet implemented for FlashAttention2Function")
import torch
import torch.nn.functional as F
import math
import argparse
import sys
import time

DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def repeat_kv(x, num_groups):
    """Helper function to repeat K/V heads for GQA naive implementation."""
    if num_groups == 1:
        return x
    B, H_kv, N, D = x.shape
    x = x.unsqueeze(2).expand(B, H_kv, num_groups, N, D)
    return x.reshape(B, H_kv * num_groups, N, D)

def create_mask_bool(
    seq_len: int,
    window_size: int,
    sink_size: int,
    device=None
    ) -> torch.Tensor:
    
    idx = torch.arange(seq_len, device=device)
    row = idx.unsqueeze(1)   # (seq_len, 1)
    col = idx.unsqueeze(0)   # (1, seq_len)

    # 1) sliding window:  i - (window_size-1) <= j <= i
    sliding = (col <= row) & (col >= row - (window_size - 1))

    # 2) sink at start:   j < sink_size  *and*  j <= i
    sink = (col < sink_size) & (col <= row)

    return sliding | sink

def naive_attention(Q, K, V, is_causal=False, window_size=None, sink_size=None):
    """
    A correct, robust PyTorch implementation of standard attention for comparison.
    Supports GQA, Sliding Window, and Attention Sinks.
    """
    
    batch_size, num_heads_q, seq_len, head_dim = Q.shape
    _, num_heads_kv, seq_len, head_dim = K.shape

    if num_heads_q != num_heads_kv:
        num_groups = num_heads_q // num_heads_kv
        K = repeat_kv(K, num_groups)
        V = repeat_kv(V, num_groups)

    scale = 1.0 / math.sqrt(head_dim)
    S = (Q @ K.transpose(-1, -2)) * scale
    
    if is_causal:
        mask = None
        if window_size is None: # Causal only
            mask = create_mask_bool(seq_len=seq_len, window_size=seq_len, sink_size=0, device=Q.device)
        else:
            if sink_size is None: # SWA only
                mask = create_mask_bool(seq_len, window_size=window_size, sink_size=0, device=Q.device)
            else: # SWA + Sink
                mask = create_mask_bool(seq_len, window_size=window_size, sink_size=sink_size, device=Q.device)
                
        S.masked_fill_(~mask, -float('inf'))

    P = torch.nn.functional.softmax(S, dim=-1, dtype=torch.float32).to(Q.dtype)
    O_final = P @ V
    L_final = torch.logsumexp(S.to(torch.float32), dim=-1)
    
    return O_final, L_final

def benchmark_attention(triton_func, naive_func, test_params, is_causal, is_gqa=False, is_swa=False):
    """Utility to benchmark an attention function and compare it to a naive implementation."""
    print("\n--- Running Performance Benchmark ---")
    window_size, sink_size = None, None
    if is_gqa and not is_swa: # GQA only 
        batch, heads_q, heads_kv, seq_len, dim = test_params
        config_str = f"B={batch}, Hq={heads_q}, Hkv={heads_kv}, L={seq_len}, D={dim}"
    elif is_swa: # GQA + SWA
        batch, heads_q, heads_kv, seq_len, dim, *window_params = test_params
        if len(window_params) == 1:
            window_size = window_params[0]
            config_str = f"B={batch}, Hq={heads_q}, Hkv={heads_kv}, L={seq_len}, D={dim}, W={window_size}"
        else:
            window_size, sink_size = window_params
            config_str = f"B={batch}, Hq={heads_q}, Hkv={heads_kv}, L={seq_len}, D={dim}, W={window_size}, S={sink_size}"
    else:
        batch, heads_q, seq_len, dim = test_params
        heads_kv = heads_q
        config_str = f"B={batch}, H={heads_q}, L={seq_len}, D={dim}"

    print(f"Benchmark Config: {config_str}, Causal={is_causal}")
    
    q = torch.randn(batch, heads_q, seq_len, dim, device='cuda', dtype=DTYPE)
    k = torch.randn(batch, heads_kv, seq_len, dim, device='cuda', dtype=DTYPE)
    v = torch.randn(batch, heads_kv, seq_len, dim, device='cuda', dtype=DTYPE)

    def _run_benchmark(func, is_triton):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        # Warm-up runs
        for _ in range(5):
            _ = func(q, k, v, is_causal=is_causal)
        
        torch.cuda.synchronize()
        start_time = time.time()
        # Timed runs
        for _ in range(20):
            _ = func(q, k, v, is_causal=is_causal)

        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time_ms = (end_time - start_time) * 1000 / 20
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        return avg_time_ms, peak_mem_gb

    triton_time, triton_mem = _run_benchmark(triton_func, is_triton=True)
    # Wrap naive func to discard the L output for benchmarking
    naive_wrapper = lambda q, k, v, is_causal: naive_func(q, k, v, is_causal=is_causal, window_size=window_size, sink_size=sink_size)[0]
    torch_time, torch_mem = _run_benchmark(naive_wrapper, is_triton=False)

    print("\n--- Benchmark Results ---")
    print(f"{'Implementation':<25} | {'Avg Time (ms)':<20} | {'Peak Memory (GB)':<20}")
    print("-" * 70)
    print(f"{'PyTorch (Naive)':<25} | {torch_time:<20.4f} | {torch_mem:<20.4f}")
    print(f"{'Triton (Flash)':<25} | {triton_time:<20.4f} | {triton_mem:<20.4f}")
    print("-" * 70)
    
    # Highlight improvements
    speedup = torch_time / triton_time if triton_time > 0 else float('inf')
    mem_saving = torch_mem / triton_mem if triton_mem > 0 else float('inf')

    print(f"Triton is {speedup:.2f}x faster than PyTorch (Naive).")
    print(f"Triton uses {mem_saving:.2f}x less memory.")

def check_problem_1():
    """Checks Problem 1: PyTorch Tiled Attention."""
    problem_num = 1
    print(f"\n--- Running Autograder for Problem {problem_num}: Tiled Flash Attention ---")
    try:
        from problem_1 import FlashAttention2Function
    except ImportError:
        print(f"Could not import FlashAttention2Function from solution_{problem_num}.py.")
        return

    torch.manual_seed(42)
    test_cases = [
        (1, 8, 512, 512, 16, False),
        (1, 8, 1024, 1024, 16, True),
        (1, 16, 2048, 2048, 16, True),
        (1, 16, 4096, 4096, 16, True),
    ]
    
    # Custom test runner for P1 which checks both O and L
    def run_p1_test(B, H, N_Q, N_K, D_H, is_causal):
        q = torch.randn(B, H, N_Q, D_H, device='cuda', dtype=DTYPE)
        k = torch.randn(B, H, N_K, D_H, device='cuda', dtype=DTYPE)
        v = torch.randn(B, H, N_K, D_H, device='cuda', dtype=DTYPE)
        
        naive_O, naive_L = naive_attention(q, k, v, is_causal=is_causal)
        student_O, student_L = FlashAttention2Function.apply(q, k, v, is_causal)
        
        o_match = torch.allclose(naive_O, student_O, rtol=5e-2, atol=5e-2)
        l_match = torch.allclose(naive_L, student_L, rtol=5e-2, atol=5e-2)
        
        param_str = f"(B={B}, H={H}, Nq={N_Q}, Nk={N_K}, D={D_H}, Causal={is_causal})"
        if o_match and l_match:
            print(f"✅ P{problem_num} Correctness Test Passed! {param_str}")
            return True
        else:
            print(f"❌ P{problem_num} Correctness Test Failed! {param_str}")
            if not o_match: print(f"   Output 'O' mismatch. Max diff: {(naive_O - student_O).abs().max()}")
            if not l_match: print(f"   Logsumexp 'L' mismatch. Max diff: {(naive_L - student_L).abs().max()}")
            return False

    results = [run_p1_test(*case) for case in test_cases]
    if all(results):
        print(f"\nAll P{problem_num} correctness tests passed!")

def check_problem_2():
    """Checks Problem 2: Triton Weighted Row-Sum."""
    problem_num = 2
    print(f"\n--- Running Autograder for Problem {problem_num}: Triton Weighted Row-Sum ---")
    try:
        from problem_2 import weighted_row_sum_forward, torch_weighted_row_sum
    except ImportError:
        print(f"Could not import functions from solution_{problem_num}.py.")
        return
        
    torch.manual_seed(43)
    test_cases = [(512, 1024), (1024, 4096), (2048, 8192), (4096, 8192)]
    
    def run_p2_test(rows, cols):
        x = torch.randn(rows, cols, device='cuda', dtype=DTYPE)
        w = torch.randn(cols, device='cuda', dtype=DTYPE)
        torch_result = torch_weighted_row_sum(x, w)
        triton_result = weighted_row_sum_forward(x, w)
        
        param_str = f"(Rows={rows}, Cols={cols})"
        if torch.allclose(torch_result, triton_result, rtol=5e-2, atol=5e-2):
            print(f"✅ P{problem_num} Correctness Test Passed! {param_str}")
            return True
        else:
            print(f"❌ P{problem_num} Correctness Test Failed! {param_str}")
            print(f" Max diff: {(torch_result - triton_result).abs().max()}")
            return False
            
    results = [run_p2_test(*case) for case in test_cases]
    if all(results): print(f"\nAll P{problem_num} correctness tests passed!")

def check_problem_3():
    """Checks Problem 3: Non-Causal Flash Attention."""
    problem_num = 3
    print(f"\n--- Running Autograder for Problem {problem_num}: Non-Causal Flash Attention ---")
    try:
        from problem_3 import flash_attention_forward
    except ImportError:
        print(f"Could not import flash_attention_forward from solution_{problem_num}.py.")
        return
        
    torch.manual_seed(44)
    test_cases = [
        (1, 8, 512, 16),
        (1, 8, 1024, 16),
        (1, 16, 2048, 16),
        (1, 16, 4096, 16),
    ]
    
    results = [run_correctness_test(case, flash_attention_forward, is_causal=False, is_gqa=False, is_swa=False, problem_num=problem_num) for case in test_cases]
    if all(results):
        print(f"\nAll P{problem_num} correctness tests passed!")
        benchmark_attention(flash_attention_forward, naive_attention, test_cases[-1], is_causal=False, is_gqa=False)

def check_problem_4():
    """Checks Problem 4: Causal Flash Attention."""
    problem_num = 4
    print(f"\n--- Running Autograder for Problem {problem_num}: Causal Flash Attention ---")
    try:
        from problem_4 import flash_attention_forward
    except ImportError:
        print(f"Could not import flash_attention_forward from solution_{problem_num}.py.")
        return
        
    torch.manual_seed(45)
    test_cases = [
        (1, 8, 512, 16),
        (1, 8, 1024, 16),
        (1, 16, 2048, 16),
        (1, 16, 4096, 16),
    ]
    
    results = [run_correctness_test(case, flash_attention_forward, is_causal=True, is_gqa=False, is_swa=False, problem_num=problem_num) for case in test_cases]
    if all(results):
        print(f"\nAll P{problem_num} correctness tests passed!")
        benchmark_attention(flash_attention_forward, naive_attention, test_cases[-1], is_causal=True, is_gqa=False)

def check_problem_5():
    """Checks Problem 5: Grouped-Query Attention."""
    problem_num = 5
    print(f"\n--- Running Autograder for Problem {problem_num}: Grouped-Query Attention ---")
    try:
        from problem_5 import flash_attention_forward
    except ImportError:
        print(f"Could not import flash_attention_forward from solution_{problem_num}.py.")
        return
        
    torch.manual_seed(46)
    # Test cases: (Batch, Heads_Q, Heads_KV, SeqLen, Dim)
    test_cases = [
        (1, 8, 2, 512, 16),
        (1, 8, 2, 1024, 16),
        (1, 16, 2, 2048, 16),
        (1, 16, 2, 4096, 16),
    ]
    
    results = [run_correctness_test(case, flash_attention_forward, is_causal=True, is_gqa=True, is_swa=False, problem_num=problem_num) for case in test_cases]
    if all(results):
        print(f"\nAll P{problem_num} correctness tests passed!")
        benchmark_attention(flash_attention_forward, naive_attention, test_cases[-1], is_causal=True, is_gqa=True)
        
def check_problem_6():
    """Checks Problem 6: Sliding Window Attention."""
    problem_num = 6
    print(f"\n--- Running Autograder for Problem {problem_num}: Sliding Window Attention ---")
    try:
        from problem_6 import flash_attention_forward
    except ImportError:
        print(f"Could not import flash_attention_forward from solution_{problem_num}.py.")
        return
        
    torch.manual_seed(47)
    # Test cases: (Batch, Heads_Q, Heads_KV, SeqLen, Dim, WindowSize)
    window_size = 128
    test_cases = [
        (1, 8, 2, 512, 16, window_size),
        (1, 8, 2, 1024, 16, window_size),
        (1, 16, 2, 2048, 16, window_size),
        (1, 16, 2, 4096, 16, window_size),
    ]
    
    results = [run_correctness_test(case, flash_attention_forward, is_causal=True, is_gqa=True, is_swa=True, problem_num=problem_num) for case in test_cases]
    if all(results):
        print(f"\nAll P{problem_num} correctness tests passed!")
        benchmark_attention(flash_attention_forward, naive_attention, test_cases[-1], is_causal=True, is_gqa=True, is_swa=True)
        

def check_problem_7():
    """Checks Problem 7: Attention Sinks."""
    problem_num = 7
    print(f"\n--- Running Autograder for Problem {problem_num}: Attention Sinks ---")
    try:
        from problem_7 import flash_attention_forward
    except ImportError:
        print(f"Could not import flash_attention_forward from solution_{problem_num}.py.")
        return
        
    torch.manual_seed(48)
    window_size, sink_size = 128, 8
    # Test cases: (Batch, Heads_Q, Heads_KV, SeqLen, Dim, Win, Sink)
    test_cases = [
        (1, 8, 2, 512, 32, window_size, sink_size),
        (1, 8, 2, 1024, 32, window_size, sink_size),
        (1, 16, 2, 2048, 16, window_size, sink_size),
        (1, 16, 2, 4096, 16, window_size, sink_size),
    ]
    
    results = [run_correctness_test(case, flash_attention_forward, is_causal=True, is_gqa=True, is_swa=True, problem_num=problem_num) for case in test_cases]
    if all(results):
        print(f"\nAll P{problem_num} correctness tests passed!")
        benchmark_attention(flash_attention_forward, naive_attention, test_cases[-1], is_causal=True, is_gqa=True, is_swa=True)
        

def run_correctness_test(test_params, student_func, is_causal, is_gqa=False, is_swa=False, problem_num=None):
    """Runs a single correctness test case for Triton implementations."""
    window_size = None
    sink_size = None

    if is_swa:
        batch, heads_q, heads_kv, seq_len, dim, *window_params = test_params
        if len(window_params) == 1:
            window_size = window_params[0]
            param_str = f"(B={batch}, Hq={heads_q}, Hkv={heads_kv}, L={seq_len}, D={dim}, W={window_size})"
        elif len(window_params) == 2:
            window_size, sink_size = window_params
            param_str = f"(B={batch}, Hq={heads_q}, Hkv={heads_kv}, L={seq_len}, D={dim}, W={window_size}, S={sink_size})"
        else:
            raise ValueError(f"Invalid window_params length: {len(window_params)}")
    elif is_gqa:
        batch, heads_q, heads_kv, seq_len, dim = test_params
        param_str = f"(B={batch}, Hq={heads_q}, Hkv={heads_kv}, L={seq_len}, D={dim})"
    else:
        batch, heads_q, seq_len, dim = test_params
        heads_kv = heads_q
        param_str = f"(B={batch}, H={heads_q}, L={seq_len}, D={dim})"

    q = torch.randn(batch, heads_q, seq_len, dim, device='cuda', dtype=DTYPE)
    k = torch.randn(batch, heads_kv, seq_len, dim, device='cuda', dtype=DTYPE)
    v = torch.randn(batch, heads_kv, seq_len, dim, device='cuda', dtype=DTYPE)
    
    torch_result, _ = naive_attention(q, k, v, is_causal=is_causal, window_size=window_size, sink_size=sink_size)
    if sink_size is not None and window_size is not None:
        triton_result = student_func(q, k, v, is_causal=is_causal, window_size=window_size, sink_size=sink_size)
    elif window_size is not None:
        triton_result = student_func(q, k, v, is_causal=is_causal, window_size=window_size)
    else:
        triton_result = student_func(q, k, v, is_causal=is_causal)

    if torch.allclose(torch_result, triton_result, rtol=5e-2, atol=5e-2):
        print(f"✅ P{problem_num} Correctness Test Passed! {param_str}")
        return True
    else:
        print(f"❌ P{problem_num} Correctness Test Failed! {param_str}")
        print(f" Max diff: {(torch_result - triton_result).abs().max()}")
        return False

    
# Main Execution

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Autograder for Triton Flash Attention assignments.")
    parser.add_argument('--p1', action='store_true', help='Run autograder for Problem 1 (PyTorch Tiled).')
    parser.add_argument('--p2', action='store_true', help='Run autograder for Problem 2 (Triton Weighted Sum).')
    parser.add_argument('--p3', action='store_true', help='Run autograder for Problem 3 (Non-Causal Flash).')
    parser.add_argument('--p4', action='store_true', help='Run autograder for Problem 4 (Causal Flash).')
    parser.add_argument('--p5', action='store_true', help='Run autograder for Problem 5 (GQA Flash).')
    parser.add_argument('--p6', action='store_true', help='Run autograder for Problem 6 (SWA Flash).')
    parser.add_argument('--p7', action='store_true', help='Run autograder for Problem 7 (Attention Sinks).')
    
    if not torch.cuda.is_available():
        print("💥 CUDA not available. Skipping all GPU tests.")
        sys.exit(1)

    args = parser.parse_args()
    if not any(vars(args).values()):
        args.p1 = args.p2 = args.p3 = args.p4 = args.p5 = args.p6 = args.p7 = True
        
    if args.p1:
        check_problem_1()
    if args.p2:
        check_problem_2()
    if args.p3:
        check_problem_3()
    if args.p4:
        check_problem_4()
    if args.p5:
        check_problem_5()
    if args.p6:
        check_problem_6()
    if args.p7:
        check_problem_7()
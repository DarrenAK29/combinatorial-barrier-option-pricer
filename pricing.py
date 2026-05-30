# pricing.py
# ----------------------------------------------------------------------
# This file implements the models derived in main.ipynb. Some terms like 'martingale' may be unfamiliar (not contained here);
# they are simply more technical, generalizable terms; we don't need to worry about them for our purposes
# ----------------------------------------------------------------------
 
from math import comb, log, ceil, sqrt
import random
 
 
def risk_neutral_p(u):
    """p = 1 / (1 + u).  From the martingale condition with r = 0"""
    return 1 / (1 + u)
 
 
def barrier_level(S0, H, u):
    """Smallest integer B such that S0 * u**B >= H"""
    if H <= S0:
        return 1
    return ceil(log(H / S0) / log(u))
 
 
def vanilla_call(N, u, p, S0, K):
    """Cox-Ross-Rubinstein price of a European call"""
    total = 0.0
    for k in range(N + 1):
        x = 2 * k - N
        payoff = max(S0 * (u ** x) - K, 0.0)
        if payoff > 0:
            total += comb(N, k) * (p ** k) * ((1 - p) ** (N - k)) * payoff
    return total
 
 
def knockout_call_brute(N, u, p, S0, K, B):
    """Up-and-out call, brute force: enumerate all 2**N paths"""
    total = 0.0
    for mask in range(1 << N):
        x = k = 0
        dead = False
        for t in range(N):
            up = (mask >> t) & 1
            x += 1 if up else -1
            k += up
            if x >= B:
                dead = True
                break
        if dead:
            continue
        payoff = max(S0 * (u ** x) - K, 0.0)
        if payoff > 0:
            total += (p ** k) * ((1 - p) ** (N - k)) * payoff
    return total
 
 
def knockout_call_reflection(N, u, p, S0, K, B):
    """Up-and-out call, reflection principle: O(N) closed form"""
    total = 0.0
    for k in range(N + 1):
        x = 2 * k - N
        if x >= B:
            continue
        payoff = max(S0 * (u ** x) - K, 0.0)
        if payoff <= 0:
            continue
        k_refl = (N + 2 * B - x) // 2
        c_total = comb(N, k)
        c_refl = comb(N, k_refl) if 0 <= k_refl <= N else 0
        n_safe = max(0, c_total - c_refl)
        if n_safe > 0:
            total += n_safe * (p ** k) * ((1 - p) ** (N - k)) * payoff
    return total
 
 
def monte_carlo_ko(N, u, p, S0, K, B, M, seed=None):
    """Up-and-out call by simulation; returns (mean, standard error)"""
    rng = random.Random(seed)
    total = total_sq = 0.0
    for _ in range(M):
        x = 0
        dead = False
        for _ in range(N):
            x += 1 if rng.random() < p else -1
            if x >= B:
                dead = True
                break
        payoff = 0.0 if dead else max(S0 * (u ** x) - K, 0.0)
        total += payoff
        total_sq += payoff * payoff
    mean = total / M
    var = max(0.0, total_sq / M - mean * mean)
    se = (var / M) ** 0.5
    return mean, se
 
 
def sample_blocked_path(N, x_end, B, max_attempts=5000, seed=None):
    """A random path with fixed endpoint x_end that touches the barrier B"""
    k = (N + x_end) // 2
    if (N + x_end) % 2 != 0 or not (0 <= k <= N):
        return None
    rng = random.Random(seed)
    for _ in range(max_attempts):
        moves = [+1] * k + [-1] * (N - k)
        rng.shuffle(moves)
        pos = 0
        for m in moves:
            pos += m
            if pos >= B:
                return moves
    return None
 
 
def reflect_path(moves, B):
    """Flip every move after the first time the path reaches B"""
    pos, hit = 0, -1
    for i, m in enumerate(moves):
        pos += m
        if pos >= B:
            hit = i
            break
    if hit < 0:
        return list(moves), -1
    reflected = list(moves)
    for i in range(hit + 1, len(reflected)):
        reflected[i] = -reflected[i]
    return reflected, hit
 
 
def run_verification(M=100_000, machine_eps=1e-10, z=1.96):
    """
    Run both cross-checks and return structured results for the notebook.
 
    Returns a dict:
      {
        "exact": [
          {"u": ..., "K": ..., "H": ...,
           "brute": ..., "refl": ..., "diff": ..., "passed": bool},
          ...
        ],
        "mc": [
          {"u": ..., "K": ..., "H": ...,
           "analytic": ..., "mc_mean": ..., "se": ...,
           "lo": ..., "hi": ..., "passed": bool},
          ...
        ],
        "exact_all_passed": bool,
        "mc_all_passed":    bool,
      }
    """
    cases_exact = [
        (20, 1.10, 100, 100, 130),
        (20, 1.08, 100,  95, 140),
        (20, 1.05, 100, 100, 120),
    ]
    cases_mc = [
        (20, 1.10, 100, 100, 130,  0),
        (20, 1.08, 100,  95, 140,  0),
        (20, 1.05, 100, 100, 120,  1),
    ]
 
    exact_rows = []
    for N, u, S0, K, H in cases_exact:
        p     = risk_neutral_p(u)
        B     = barrier_level(S0, H, u)
        brute = knockout_call_brute(N, u, p, S0, K, B)
        refl  = knockout_call_reflection(N, u, p, S0, K, B)
        diff  = abs(brute - refl)
        exact_rows.append({
            "u": u, "K": K, "H": H,
            "brute": brute, "refl": refl,
            "diff": diff, "passed": diff < machine_eps,
        })
 
    mc_rows = []
    for N, u, S0, K, H, seed in cases_mc:
        p        = risk_neutral_p(u)
        B        = barrier_level(S0, H, u)
        analytic = knockout_call_reflection(N, u, p, S0, K, B)
        mean, se = monte_carlo_ko(N, u, p, S0, K, B, M, seed=seed)
        lo, hi   = mean - z * se, mean + z * se
        mc_rows.append({
            "u": u, "K": K, "H": H,
            "analytic": analytic, "mc_mean": mean, "se": se,
            "lo": lo, "hi": hi, "passed": lo <= analytic <= hi,
        })
 
    return {
        "exact":            exact_rows,
        "mc":               mc_rows,
        "exact_all_passed": all(r["passed"] for r in exact_rows),
        "mc_all_passed":    all(r["passed"] for r in mc_rows),
    }
 
 
def verify():
    """from pricing import verify; verify()"""
    cases = [
        (20, 1.10, 100, 100, 130, 0),
        (20, 1.08, 100,  95, 140, 0),
        (20, 1.05, 100, 100, 120, 1),
    ]
    M = 100_000
    Z = 1.96
 
    header = "  u      K    H      brute       refl      |diff|    MC mean       95% CI              result"
    print(header)
    print("-" * (len(header)))
 
    for N, u, S0, K, H, seed in cases:
        p        = risk_neutral_p(u)
        B        = barrier_level(S0, H, u)
        brute    = knockout_call_brute(N, u, p, S0, K, B)
        refl     = knockout_call_reflection(N, u, p, S0, K, B)
        mean, se = monte_carlo_ko(N, u, p, S0, K, B, M, seed=seed)
        lo       = mean - Z * se
        hi       = mean + Z * se
        diff     = abs(brute - refl)
        ok       = diff < 1e-10 and lo <= refl <= hi
 
        col_u      = str(u).rjust(4)
        col_K      = str(K).rjust(4)
        col_H      = str(H).rjust(4)
        col_brute  = f"{brute:.6f}".rjust(10)
        col_refl   = f"{refl:.6f}".rjust(10)
        col_diff   = f"{diff:.2e}".rjust(9)
        col_mean   = f"{mean:.6f}".rjust(10)
        col_ci     = f"[{lo:.4f}, {hi:.4f}]".ljust(18)
        col_result = "PASS" if ok else "FAIL"
 
        print(f"  {col_u}  {col_K}  {col_H}  {col_brute}  {col_refl}  {col_diff}  {col_mean}  {col_ci}  {col_result}")

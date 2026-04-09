# For N > this threshold, residual() and neg_log_likelihood() use chunked
# computation to avoid materialising the full N×N matrix.
DCM_LARGE_N_THRESHOLD: int = 5_000
DWCM_LARGE_N_THRESHOLD: int = 5_000
DaECM_LARGE_N_THRESHOLD: int = 2_000

# Number of rows processed per chunk when using memory-efficient mode.
_DEFAULT_CHUNK: int = 512

# θ lower bound: ensures β = exp(−θ) < 1 and avoids div-by-zero in w_ij.
_ETA_MIN: float = 1e-10
# θ upper bound: exp(−50) ≈ 2e-22, essentially zero weight contribution.
_ETA_MAX: float = 50.0

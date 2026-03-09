# Istruzioni per Copilot — Progetto MaxEnt Network Solver

## Linguaggio e stack
- Python 3.10+
- PyTorch per il calcolo tensoriale (GPU support opzionale)
- NumPy/SciPy come fallback o per operazioni sparse
- Numba per JIT-compilazione di hot loops
- `tracemalloc` e `time.perf_counter` per profiling

## Convenzioni di codice
- Type hints obbligatori su tutte le funzioni pubbliche
- Docstring Google-style
- Precisione: usare sempre `float64` / `torch.float64` salvo indicazione contraria
- Nomi variabili: seguire la notazione dei paper (x_i, y_j per i moltiplicatori di Lagrange, k_out, k_in, s_out, s_in per le sequenze osservate)
- Ogni solver deve restituire un dataclass `SolverResult` con: soluzione, flag convergenza, numero iterazioni, storia residui, tempo, picco RAM

## Testing
- Ogni metodo numerico deve avere test su sistemi piccoli (n=4,10) con soluzione nota
- Tolleranza convergenza test: errore massimo sui vincoli < 1e-6
- Usare `pytest`

## Performance
- Per reti grandi (n > 10k), evitare la materializzazione di matrici NxN dense quando possibile
- Sfruttare la degree reduction (nodi con stesso grado condividono lo stesso moltiplicatore)
- Profilare RAM e tempo per ogni metodo e riportare i risultati
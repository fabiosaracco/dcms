# Piano operativo per l'agente

## Contesto del progetto

Questo progetto risolve numericamente le equazioni di massima entropia per modelli di reti complesse dirette.
I modelli da implementare sono quattro, in ordine crescente di complessità:

### Modello 1: DCM (Directed Configuration Model) — binario
- Vincoli: sequenza di gradi in/out (k_out_i, k_in_i) per ogni nodo i
- Incognite: 2N moltiplicatori di Lagrange (x_i, y_i), uno per out-degree e uno per in-degree
- Equazioni (nella parametrizzazione esponenziale, con θ tali che x=exp(-θ)):
  ```
  k_out_i = Σ_{j≠i} (x_i * y_j) / (1 + x_i * y_j)
  k_in_i  = Σ_{j≠i} (x_j * y_i) / (1 + x_j * y_i)
  ```
- Dimensione del sistema: 2N equazioni in 2N incognite
- Con degree reduction: il numero di incognite scende al numero di valori distinti di (k_out, k_in)
- **Status: ✅ COMPLETATO** — converge al 100% fino a N=50k con FP-GS

### Modello 2: DWCM (Directed Weighted Configuration Model) — pesato discreto
- Vincoli: sequenza di strengths in/out (s_out_i, s_in_i)
- Incognite: 2N moltiplicatori (β_out_i, β_in_i)
- Equazioni (pesi interi, distribuzione geometrica):
  ```
  s_out_i = Σ_{j≠i} (β_out_i * β_in_j) / (1 - β_out_i * β_in_j)
  s_in_i  = Σ_{j≠i} (β_out_j * β_in_i) / (1 - β_out_j * β_in_i)
  ```
- ATTENZIONE: richiede β_out_i * β_in_j < 1 per ogni coppia (i,j) — vincolo di feasibility
- Dimensione: 2N equazioni
- Con strength reduction: il numero di incognite scende al numero di valori distinti di (s_out, s_in)
- **Status: ✅ COMPLETATO** — converge con θ-Newton Anderson(10) fino a N=10k.

### Modello 3: qDECM (Quasi Directed Enhanced Configuration Model) — binario + pesato
- Vincoli: sequenza di gradi E strengths (k_out_i, k_in_i, s_out_i, s_in_i)
- Incognite: 4N moltiplicatori (x_i, y_i, β_out_i, β_in_i)
- Equazioni:
  ```
  k_out_i = Σ_{j≠i} (x_i * y_j) / (1 + x_i * y_j)
  k_in_i  = Σ_{j≠i} (x_j * y_i) / (1 + x_j * y_i) [come nel DCM]
  s_out_i = Σ_{j≠i} p_ij / (1 - β_out_i · β_in_j)
  s_in_i  = analoga  [E[w_ij] = p_ij · 1/(1-z_ij), non p_ij · z_ij/(1-z_ij)]
  ```
- Dimensione: 4N equazioni in 4N incognite
- In pratica, prima si risolve il DCM (riusare le routine esistenti), poi si risolve un DWCM condizionato alla topologia del DCM.
- È una versione approssimata del DECM con più ampie possibilità di convergenza.
- **Status: ✅ COMPLETATO** — two-step solver (DCM → conditioned DWCM), benchmark N=5k

### Modello 4: DECM (Directed Enhanced Configuration Model) — binario + pesato
- Vincoli: sequenza di gradi E strengths (k_out_i, k_in_i, s_out_i, s_in_i)
- Incognite: 4N moltiplicatori (θ_out_i, θ_in_i, η_out_i, η_in_i) con x_i=e^{-θ}, β_i=e^{-η}
- Equazioni (accoppiate — η entra in p_ij):
  ```
  p_ij    = sigmoid(-θ_out_i - θ_in_j - log(expm1(η_out_i + η_in_j)))
  G_ij    = 1 / (1 - β_out_i * β_in_j)      [fattore peso, β = e^{-η}]
  k_out_i = Σ_{j≠i} p_ij
  k_in_i  = analoga
  s_out_i = Σ_{j≠i} p_ij * G_ij             [E[w_ij] = p_ij / (1 - β_out_i β_in_j)]
  s_in_i  = analoga
  ```
- Dimensione: 4N equazioni in 4N incognite
- Questo è il modello più difficile da far convergere: le equazioni di grado e strength sono **accoppiate** (η entra in p_ij)
- **Status: ✅ COMPLETATO** — alternating GS-Newton Anderson(10), 100% convergenza fino a N=5k

## Metodi di risoluzione

> **Lezione appresa dalle Fasi 1-5:** solo i metodi Fixed-Point scalano e convergono in modo affidabile per N grande. I metodi Newton-like (Newton pieno, Broyden, LM) richiedono O(N²) RAM e non scalano oltre N≈500. L-BFGS scala in RAM ma non converge in modo affidabile sul DWCM puro per reti eterogenee; tuttavia funziona sul qDECM con warm-start dalla soluzione two-step. Pertanto, il DECM va implementato usando **prioritariamente** i metodi seguenti.

### Metodo 1: Fixed-Point Gauss-Seidel (β-space)
- Aggiorna ciascun moltiplicatore isolandolo dalla rispettiva equazione
- Ordine Gauss-Seidel: aggiorna prima θ_out (o x), poi θ_in (o y) usando i valori appena aggiornati
- Convergenza lineare, O(N) per iterazione, RAM O(N)
- Damping opzionale α ∈ (0, 1] — usare α=1.0 di default, ridurre se oscilla
- **Anderson acceleration** (depth 5-10) per accelerare la convergenza: mantiene una storia dei residui e usa mixing lineare per predire il passo successivo
- Multi-start con 4 inizializzazioni: "strengths"/"degrees", "normalized", "uniform", "random"
- **Dove funziona:** DCM a tutte le scale; DWCM per reti non troppo eterogenee; qDECM (parte pesata)
- **Dove fallisce:** DWCM con nodi hub (β → 1), causa oscillazioni

### Metodo 2: θ-Newton coordinato (θ-space)
- Riscrittura del fixed-point in θ-space: per ogni nodo i, Newton 1D sull'equazione
  ```
  F_i(θ) = Σ_{j≠i} 1/expm1(θ_out_i + θ_in_j) - s_out_i = 0
  ```
- Step di Newton per nodo: Δθ_i = -F_i / F'_i, clippato a ±max_step
- Non può mai produrre β > 1 (perché θ resta sempre > 0)
- Ordine Gauss-Seidel: aggiorna θ_out prima, poi θ_in con valori freschi
- **Anderson acceleration** (depth 10) per convergenza superlineare
- Multi-start con 4 inizializzazioni
- **Dove funziona:** DWCM a tutte le scale, incluse reti con hub estremi; qDECM (parte pesata)
- **Questo è il metodo di riferimento per DWCM e probabilmente per DECM**

### Metodo 3: Two-step solver (solo qDECM)
- **Step 1:** risolvere il DCM con FP-GS → ottenere (x*, y*)
- **Step 2:** con p_ij fissati dal DCM, risolvere la parte pesata con θ-Newton Anderson(10)
- Codice: `src/solvers/qdecm_solver.py`
- Funziona perché il qDECM è separabile: le equazioni di grado non dipendono da β
- **Non applicabile al DECM** (dove le equazioni di k dipendono da β)

### Dettagli implementativi comuni
- Chunked computation per N > 5000: non materializzare mai la matrice N×N completa, calcolare a blocchi di chunk×N
- **Backend selezionabile**: ogni solver accetta un parametro `backend` (`"auto"`, `"pytorch"`, `"numba"`).
  - `"auto"` (default): PyTorch chunked per N ≤ 100 000, Numba scalar per N > 100 000 (soglia aggiornata al 2026-07-13 in base a benchmark reali su DECM N=50k/100k/200k su stella: RAM di picco praticamente identica tra i due backend a tutte le scale — Numba NON offre vantaggio di RAM come si credeva in origine; PyTorch è più veloce fino a N≈50k, Numba diventa ~13-14% più veloce solo da N≥100k)
  - `"pytorch"`: forza PyTorch (dense o chunked a seconda di N)
  - `"numba"`: forza Numba JIT-compilato
  - Fallback automatico con warning se il backend richiesto non è disponibile
  - Numba è opzionale: `pip install dcms[numba]`
  - Nota: Numba non è necessario per supportare Python < 3.9 — PyTorch da solo gira correttamente su Python 3.8 (testato); il vincolo `requires-python>=3.9` in `pyproject.toml` è una scelta di packaging, non un limite tecnico legato al backend
- Zero-strength/zero-degree nodes: fissare θ = θ_MAX e non aggiornare mai
- Convergenza: ‖F‖∞ < tol (default 1e-5)
- Profiling: ogni solver restituisce `SolverResult` con theta, converged, iterations, residuals, elapsed_time, peak_ram_bytes

## Piano di lavoro — SEGUI QUESTO ORDINE

### Fase 1: Infrastruttura ✅ COMPLETATA
1. Struttura del progetto: `src/models/`, `src/solvers/`, `src/benchmarks/`, `src/utils/`, `tests/`
2. `SolverResult` dataclass con profiling
3. Degree/strength reduction
4. Generatore reti di test: `src/utils/wng.py` → `k_s_generator_pl`

### Fase 2: DCM ✅ COMPLETATA
- Equazioni, gradiente, Hessiano diagonale
- FP-GS converge al 100% su tutte le taglie testate (N=10 … 50k)
- Codice: `src/models/dcm.py`, `src/solvers/fixed_point.py`

### Fase 3: Scaling DCM ✅ COMPLETATA
- Testato fino a N=50k
- FP-GS α=1.0: il più veloce, < 10 iterazioni tipicamente
- Benchmark: `src/benchmarks/dcm_comparison.py`, `src/benchmarks/dcm_scaling.py`

### Fase 4: DWCM ✅ COMPLETATA (fino a N=5k, N=10k in corso)
- Equazioni e modello: `src/models/dwcm.py`
- FP-GS β-space: funziona ma oscilla su reti con hub (β → 1)
- θ-Newton Anderson(10): risolve il problema hub, converge su tutti i seed testati a N=5k
- Benchmark: `src/benchmarks/dwcm_comparison.py` (flag `--phase4`, `--sizes`, `--fast`)
- Codice: `src/solvers/fixed_point_dwcm.py` (varianti "gauss-seidel", "theta-newton")

### Fase 5: qDECM ✅ COMPLETATA
- Modello: `src/models/qdecm.py` — equazioni condizionate W_ij = p_ij / (1 − β_out_i · β_in_j)
- Two-step solver: `src/solvers/qdecm_solver.py` — DCM → conditioned DWCM (θ-Newton Anderson)
- FP solver per la parte pesata: `src/solvers/fixed_point_qdecm.py` — varianti FP-GS, Jacobi, θ-Newton + Anderson
- Metodi joint su 4N variabili: `residual_joint`, `neg_log_likelihood_joint`, `constraint_error_joint`
- 46 unit test: `tests/test_qdecm.py`
- Benchmark N=1k: `src/benchmarks/qdecm_comparison.py`
- README aggiornato con sezione qDECM (modello 1.3 + tabella N=1k)

### Fase 6: DECM ✅ COMPLETATA
- Modello: `dcms/models/decm.py` — equazioni accoppiate (η entra in p_ij)
- Solver: `dcms/solvers/fixed_point_decm.py` — alternating GS-Newton Anderson(10) su 4N variabili
- Multi-start: se "degrees" IC non converge, retry con "qdecm" warm-start e "random"
- Hub bisection: `hub_sk_threshold` per nodi con s/k >> 1 (stessa logica del qDECM)
- Benchmark N=1k e N=5k: 100% convergenza su tutti i seed testati
- Test: `tests/test_decm.py`

### Fase 7: Hub bisection ✅ COMPLETATA
- Problema: nodi con s_i/k_i >> 1 causano β_i → 1 e instabilità numerica nel global Newton step
- Soluzione: per i nodi "hub" (s/k > threshold), trovare η esattamente via bisection 1D per iterazione
- qDECM (file: `dcms/solvers/fixed_point_qdecm.py`): bisection su β ∈ [0, 1/max(β_other)); funzione crescente
- DECM (file: `dcms/solvers/fixed_point_decm.py`): bisection su η ∈ [_ETA_MIN, _ETA_MAX]; funzione decrescente
- 3 sweep GS per iterazione garantiscono consistenza se un nodo è sia out-hub che in-hub
- Anderson: r_k zerate per componenti hub (esclusione); θ_next ripristinato a valori bisection (freeze)
- Parametro: `hub_sk_threshold=0.0` (default, disabilitato) in `solve_tool()` per qDECM e DECM
- Test: quirinale_dico4 (N=22754): best_MRE=9.5e-5 con threshold=5 in 46 iter (~8 min)
3. Raccomandazione finale: quale metodo per quale modello a quale scala

## Lezioni apprese (Fasi 1-5)

Queste informazioni servono per evitare di ripetere esperimenti falliti:

| Metodo | DCM | DWCM | qDECM | Motivo del fallimento |
|--------|-----|------|-------|----------------------|
| FP-GS α=1.0 | ✅ 100% | ⚠ oscilla su hub | ✅ (parte pesata) | β > 1 causa periodo-3 su DWCM |
| FP-GS α=0.3-0.5 | ✅ più lento | ⚠ stesso problema | ✅ | damping non basta per DWCM hub |
| FP-GS Anderson(10) | ✅ | ⚠ migliora ma non risolve hub | ✅ | mixing history contaminata su DWCM |
| θ-Newton Anderson(10) | N/A | ✅ 100% fino a N=5k | ✅ | **metodo di riferimento per peso** |
| Two-step (DCM→DWCM cond.) | N/A | N/A | ✅ | separabile, robusto |
| L-BFGS (2N) | ✅ DCM | ❌ DWCM eterogeneo | N/A | non converge su reti power-law |
| Newton pieno | ✅ N≤500 | ✅ N≤500 | ✅ N≤500 | O(N²) RAM, non scala |
| Broyden | ✅ N≤500 | ⚠ instabile | N/A | rank-1 update diverge |
| LM (full Jacobian) | ✅ N≤500 | ✅ N≤500 | ✅ N≤500 | O(N²) RAM, non scala |
| LM (diag Hessian) | ⚠ lento | ❌ | ⚠ | non converge affidabilmente |

**Conclusione: usare FP-GS + θ-Newton con Anderson per la parte pesata. Per il DECM, provare prima la strategia alternata (FP-GS topo + θ-Newton peso) con warm-start da qDECM.**

## Criteri di successo
- Un metodo "converge" se ‖F(θ)‖∞ < 1e-5
- Per reti da 10k nodi, almeno un metodo deve convergere in meno di 15 minuti
- Il codice deve essere testabile con `pytest` su sistemi piccoli (CI-friendly, < 30 sec)

## Riferimenti implementativi
- Paper di riferimento: Squartini & Garlaschelli, New J. Phys. 13 (2011) 083001
- Implementazione di riferimento: NEMtropy (https://github.com/nicoloval/NEMtropy)
- Anderson acceleration: Walker & Ni, SIAM J. Numer. Anal. 49(4), 2011

## Cosa NON fare
- NON usare scipy.optimize.fsolve o root come black-box — vogliamo controllare il solver
- NON materializzare matrici NxN dense per N > 5000
- NON implementare Newton pieno, Broyden, o LM full-Jacobian per il DECM a scala grande — non scalano e abbiamo già verificato
- NON dichiarare un metodo "non funzionante" dopo un singolo tentativo — provare almeno 4 inizializzazioni diverse
- NON ignorare i NaN — se compaiono, loggare dove e con quali parametri e fermarsi con grazia
- NON riscrivere le parti DCM/DWCM/qDECM del README quando aggiungi il DECM — solo appendere

## Generazione reti di test

Il codice per generare reti sintetiche è in `src/utils/wng.py`. La funzione è `k_s_generator_pl`.

```python
k, s = k_s_generator_pl(N, rho=1e-3, seed=42)
# k: tensor (2N,) = [k_out | k_in]
# s: tensor (2N,) = [s_out | s_in]
```

#### Logica del test (IMPORTANTE):
Il solver ha successo se i vincoli ricostruiti dai θ_trovati coincidono con quelli osservati.

#### Taglie di rete per i test:
| Scopo | N nodi | Dove usarlo |
|---|---|---|
| Unit test (CI) | 10-50 | `tests/`, deve girare in < 5 sec |
| Validazione | 100-1000 | verifica convergenza |
| Scaling | 5k, 10k | benchmark RAM/tempo |

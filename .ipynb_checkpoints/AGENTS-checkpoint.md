# Piano operativo per l'agente

## Contesto del progetto

Questo progetto risolve numericamente le equazioni di massima entropia per modelli di reti complesse dirette.
I modelli da implementare sono tre, in ordine crescente di complessità:

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

### Modello 3: DaECM (Directed approximated Enhanced Configuration Model) — binario + pesato
- Vincoli: sequenza di gradi E strengths (k_out_i, k_in_i, s_out_i, s_in_i)
- Incognite: 4N moltiplicatori (x_i, y_i, β_out_i, β_in_i)
- Equazioni:
  ```
  k_out_i = Σ_{j≠i} (x_i * y_j) / (1 + x_i * y_j)
  k_in_i  = Σ_{j≠i} (x_j * y_i) / (1 + x_j * y_i) [come nel DCM]
  s_out_i = Σ_{j≠i} (x_j * y_i) / (1 + x_j * y_i)/ (1 - β_out_j * β_in_i)
 [valore atteso del peso condizionato alla topologia]
  s_in_i  = analoga
  ```
- Dimensione: 4N equazioni in 4N incognite
- In pratica, prima si risolve il DCM (quindi tutte le routine per il DCM possono essere richiamate e riusate), poi si risolve un DWCM condizionato alla topologia del DCM.
- È una versione approssimata del modello successivo e con più ampie possibilità di convergenza. 


### Modello 4: DECM (Directed Enhanced Configuration Model) — binario + pesato
- Vincoli: sequenza di gradi E strengths (k_out_i, k_in_i, s_out_i, s_in_i)
- Incognite: 4N moltiplicatori (x_i, y_i, β_out_i, β_in_i)
- Equazioni:
  ```
  k_out_i = Σ_{j≠i} (x_i * y_j) / (1 + x_i * y_j * (1/(1 - β_out_i * β_in_j) - 1))   [formula completa dal paper]
  k_in_i  = analoga
  s_out_i = Σ_{j≠i} [valore atteso del peso condizionato alla topologia]
  s_in_i  = analoga
  ```
- Dimensione: 4N equazioni in 4N incognite
- Questo è il modello più difficile da far convergere

## Metodi di risoluzione da implementare

Implementa TUTTI i seguenti metodi. L'ordine riflette una priorità ragionevole:

### Metodo 1: Fixed-Point Iteration
- Il più semplice, baseline obbligatorio
- Aggiorna ciascun moltiplicatore isolandolo dalla rispettiva equazione
- Convergenza lineare, nessuna matrice da invertire → O(N) per iterazione, RAM O(N)
- Può non convergere o convergere lentamente per reti eterogenee
- Implementare con e senza damping (α ∈ [0.1, 0.5, 1.0])
- In alcuni casi testati l'approccio a la Jacobi (ovvero, aggiornare tutte le fitness tutte insieme) ha mostrato performance peggiori del Gauss-Siedel
(ovvero l'aggiornamento) all'interno del solito step, prima delle fitness 'in', per esempio, poi delle fitness 'out', usando le fitness 'in' aggiornate. Prova entrambi i casi. 


### Metodo 2: Quasi-Newton (L-BFGS o BFGS diagonale)
- Usa il gradiente della log-likelihood (che è = al residuo del sistema)
- Approssimazione dell'Hessiano: diagonale o L-BFGS con m=10-20 vettori
- Line search con condizioni di Wolfe
- RAM: O(N*m) per L-BFGS, O(N) per diag BFGS
- Questo è il metodo che NEMtropy usa come default ed è generalmente il migliore

### Metodo 3: Newton pieno
- Jacobiano esatto + risoluzione del sistema lineare J·δθ = -F(θ)
- Convergenza quadratica ma O(N²) in RAM e O(N³) per iterazione
- Usabile solo per N < ~5000
- Implementare regolarizzazione del Jacobiano (Tikhonov: J + εI)
- Se possibile, usare `torch.linalg.solve` anziché inversione esplicita

### Metodo 4: Newton con Jacobiano approssimato (Broyden)
- Calcolare il Jacobiano solo al primo step, poi aggiornamenti rank-1
- Sherman-Morrison per aggiornare J_inv
- Convergenza superlineare
- Stesso costo RAM di Newton (O(N²)) ma costo/iterazione O(N²) anziché O(N³)

### Metodo 5: Levenberg-Marquardt
- Newton regolarizzato: (J'J + λI)δθ = -J'F(θ)
- λ adattivo: aumenta se il passo peggiora, diminuisce se migliora
- Il più robusto dei metodi Newton-like
- Per N grande, usa solo l'Hessiano diagonale

## Piano di lavoro — SEGUI QUESTO ORDINE

### Fase 1: Infrastruttura (fare per prima)
1. Creare la struttura del progetto:
   ```
   src/
     models/        → definizioni delle equazioni (DCM, DWCM, DECM)
     solvers/       → implementazioni dei metodi
     benchmarks/    → script di confronto
     utils/         → profiling, degree reduction, I/O
   tests/
   ```
2. Implementare il dataclass `SolverResult` e il wrapper di profiling (tempo + RAM)
3. Implementare la degree reduction: raggruppare nodi con stessi vincoli per ridurre la dimensione del sistema
4. Implementare la generazione di grafi di test con soluzioni note (forward problem: dai parametri genera il grafo, poi risolvi l'inverso)

### Fase 2: DCM (il modello più semplice)
1. Implementare le equazioni del DCM (forward: parametri → valori attesi; backward: sistema da risolvere)
2. Implementare il gradiente e l'Hessiano diagonale della log-likelihood del DCM
3. Implementare TUTTI e 5 i metodi di risoluzione per il DCM
4. Testare su grafi piccoli (n=10, 50, 100) verificando che la soluzione ricostruisca i gradi osservati
5. Se un metodo non converge, analizzare il residuo e provare condizioni iniziali diverse
6. Confrontare i metodi: tabella con (metodo, converge?, iterazioni, tempo, RAM)

### Fase 3: Scaling del DCM
1. Testare i metodi su n=1000, 5000, 10000, 50000
2. Per n > 5000, il Newton pieno probabilmente non ci sta in RAM — verificare e documentare
3. Identificare il metodo migliore (rapporto convergenza/tempo/RAM) a ciascuna scala
4. Ottimizzare: vettorizzazione PyTorch, eventuale GPU, eventuale sparse ops

### Fase 4: DWCM
1. Implementare le equazioni del DWCM
2. ATTENZIONE al vincolo β_out_i * β_in_j < 1 — clampare i parametri ad ogni iterazione
3. Implementare i metodi che hanno funzionato per il DCM
4. Testare e confrontare come nelle Fasi 2-3

### Fase 5: DaECM
1. Implementare le equazioni del DaECM
2. usare il DCM per la parte topologica
3. ATTENZIONE al vincolo β_out_i * β_in_j < 1 — clampare i parametri ad ogni iterazione
4. Implementare i metodi che hanno funzionato per il DCM
5. Testare e confrontare come nelle Fasi 2-3

### Fase 6: DECM
1. Implementare le equazioni del DECM (4N incognite — il sistema più grande)
2. Le equazioni sono accoppiate tra parte topologica e parte pesata — convergenza più difficile
3. Strategia: usare la soluzione DCM come guess iniziale per la parte topologica
4. Implementare i metodi, testare, confrontare

### Fase 7: Report finale
1. Tabella comparativa finale per tutti i modelli e tutti i metodi
2. Per ogni combinazione (modello, metodo, dimensione): convergenza, iterazioni, tempo, RAM
3. Raccomandazione: quale metodo usare per quale modello a quale scala
4. Identificare se servono metodi aggiuntivi (es. trust-region, Anderson acceleration)

## Criteri di successo
- Un metodo "converge" se il max errore assoluto sui vincoli è < 1e-5
- Per reti da 50k nodi, almeno un metodo deve convergere in meno di 5 minuti
- Il codice deve essere testabile con `pytest` su sistemi piccoli (CI-friendly, < 30 sec)

## Riferimenti implementativi
- Paper di riferimento: Squartini & Garlaschelli, New J. Phys. 13 (2011) 083001
- Implementazione di riferimento: NEMtropy (https://github.com/nicoloval/NEMtropy)
  - Il loro solver usa: fixed-point, newton, quasinewton con line search e Wolfe conditions
  - Parametrizzazione esponenziale (θ-space)
  - Regolarizzazione dell'Hessiano quando necessario

## Cosa NON fare
- NON usare scipy.optimize.fsolve o root come black-box — vogliamo controllare il solver
- NON materializzare matrici NxN dense per N > 10000 (a meno che non sia il Newton pieno, che va limitato a N piccoli)
- NON dichiarare un metodo "non funzionante" dopo un singolo tentativo — provare almeno 5 condizioni iniziali diverse
- NON ignorare i NaN — se compaiono, loggare dove e con quali parametri e fermarsi con grazia

## Generazione reti di test

Il codice per generare reti sintetiche con soluzioni note è in `src/utils/wng.py`. La funzione di cui hai bisogno è k_s_generator_pl.

Usalo così:genera una rete diretta a partire da parametri di default, poi usa le sequenze di grado e di strenght  risultanti come input del solver. 


#### Logica del test (IMPORTANTE):
Il solver ha successo se i vincoli ricostruiti dai θ_trovati coincidono con quelli osservati.

#### Taglie di rete per i test:
| Scopo | N nodi | Dove usarlo |
|---|---|---|
| Unit test (CI) | 10-50 | `tests/`, deve girare in < 5 sec |
| Validazione | 100-1000 | Fase 2-4, verifica convergenza |
| Scaling | 5k, 10k, 50k | Fase 3-5, benchmark RAM/tempo |

---
applyTo: "src/solvers/**/*.py"
---

# Istruzioni per i file solver

## Parametrizzazione
Tutti i moltiplicatori di Lagrange devono essere nella parametrizzazione esponenziale:
- Variabili interne: θ_i (log-space)
- Variabili fisiche: x_i = exp(-θ_i)
- Questo garantisce positività senza vincoli e migliore stabilità numerica

## Struttura di ogni solver
Ogni solver deve:
1. Accettare: funzione sistema F(θ), Jacobiano (o approssimazione), guess iniziale θ0, tolleranza, max iterazioni
2. Implementare line search (Wolfe conditions) quando applicabile
3. Loggare la storia di convergenza (norma residuo ad ogni iterazione)
4. Gestire gracefully la non-convergenza (restituire il miglior risultato parziale)
5. Misurare tempo e picco RAM tramite il decorator/wrapper di profiling

## Robustezza
- Se il Jacobiano è singolare o mal condizionato, applicare regolarizzazione (Tikhonov o eigenvalue-based)
- Se un metodo diverge, provare damping progressivo prima di dichiarare fallimento
- Clampare i valori di θ per evitare overflow/underflow: θ ∈ [-50, 50]
# Cloud-based MCDM core System

## 1. data_loader.py

**Purpose:** Load BWM input (Best-to-Others, Others-to-Worst) from a single Excel file and infer dimensions.

**Role in flow:** First step. Called by `main.py`; output feeds `interval_bwm`.

**Key function**

| Signature | Responsibility |
|-----------|----------------|
| `load_bwm_data_from_excel(filepath: str) -> (BO, OW, K, C, A)` | Read Excel, infer K/C/A from layout, return nested lists BO[k][j][i], OW[k][j][i] and scalars K, C, A. |

**Step-by-step logic (pseudocode)**

```
1. df = pd.read_excel(filepath, header=None)
2. alt_row = df.iloc[1, 2:].dropna()   # row 2, skip cols 0–1
3. A = 1; while alt_row[A] != alt_row[0]: A += 1
4. C = len(alt_row) // A
5. K = (len(df) - 2) // 2
6. BO, OW = [], []
7. for k in 0..K-1:
8.     bo_row = 2 + k*2, ow_row = 2 + k*2 + 1
9.     for j in 0..C-1: start = 2 + j*A, end = start + A
10.        append df[bo_row, start:end] to BO[k][j], df[ow_row, start:end] to OW[k][j]
11. return BO, OW, K, C, A
```

**Inputs**

- **Args:** `filepath` — path to `.xlsx`.
- **Files:** One Excel file (rows 1–2 headers; then K pairs of BO/OW data rows).
- **Config/env:** None.

**Outputs / side effects**

- **Returns:** `BO`, `OW` (list of list of lists of float), `K`, `C`, `A` (int). No files written, no global state.

**Dependencies**

- `pandas`, `numpy`. Requires `openpyxl` for `.xlsx`.

**Edge cases / failure modes**

- Missing file → pandas/OS error.
- Row 2 too short or no repeat of first alternative label → A or C wrong; possible index errors or wrong parsing.
- `(len(df)-2)` not even → K fractional; integer division still used (logic error).
- No explicit check that row 1/2 match expected header/label format.

---

## 2. bwm_solver.py

**Purpose:** Solve a single BWM problem (one decision maker, one criterion): exact weights and lower/upper bounds under consistency constraints.

**Role in flow:** Not called from `main.py` directly. Used by `interval_bwm.py` for each (k, j); also by `main.py` to recompute exact weights for display.

**Key class and methods**

| Signature | Responsibility |
|-----------|----------------|
| `BWM_Solver_SciPy(best_to_others, others_to_worst, best_index=None, worst_index=None)` | Store BO, OW; validate; set B, W (best/worst indices). |
| `_validate_and_detect_indices() -> (best_idx, worst_idx)` | Require exactly one 1 in BO and one 1 in OW; require BO, OW ≥ 1; return indices of those 1s. Raise ValueError on violation. |
| `solve_exact() -> (w, e_star)` | Minimize e s.t. e ≥ \|w_B/w_i - BO_i\|, e ≥ \|w_i/w_W - OW_i\|, Σw=1, w≥0. SLSQP. Set self.weights, self.e_star. |
| `solve_lower_bounds() -> L` | For each t: minimize w_t s.t. same BWM constraints at e_star; clip/validate; enforce L ≤ weights. Set self.lower. |
| `solve_upper_bounds() -> U` | For each t: maximize w_t (min -w_t) s.t. same constraints; clip/validate; enforce U ≥ lower, U ≥ weights. Set self.upper. |
| `solve_all() -> {weights, e_star, lower, upper}` | Call solve_exact, then solve_lower_bounds, then solve_upper_bounds; return dict. |

**Step-by-step logic (solve_all)**

```
1. solve_exact:
   - x = [w_1..w_n, e]. Minimize e.
   - ineq: e - |w_B/w_i - BO_i| ≥ 0, e - |w_i/w_W - OW_i| ≥ 0 for all i.
   - eq: sum(w) = 1. Bounds w_i ≥ 1e-6, e ≥ 0.
   - Clip w to non-negative, renormalize; set self.weights, self.e_star.

2. solve_lower_bounds:
   - For t = 0..n-1: minimize w_t s.t. same ineq at e_star, sum(w)=1.
   - If not res.success: L[t] = fallback (e.g. max(0, weights[t]-0.1)).
   - Else clip to [0,1]; then if L[t] > weights[t] set L[t] = weights[t].
   - Set self.lower.

3. solve_upper_bounds:
   - For t = 0..n-1: minimize -w_t s.t. same ineq, sum(w)=1.
   - If not res.success: U[t] = fallback.
   - Else value = -res.fun; clip to [0,1]; then U[t] ≥ lower[t], U[t] ≥ weights[t].
   - Set self.upper.
```

**Inputs**

- **Args:** `best_to_others`, `others_to_worst` (array-like length n); optional `best_index`, `worst_index`.
- **Config:** None. Optimizer options hardcoded (maxiter, ftol).

**Outputs / side effects**

- **Returns:** From `solve_all`: dict with `weights`, `e_star`, `lower`, `upper`. Instance attributes updated.
- **Side effects:** Prints warnings on optimization failure or bound violations; no files.

**Dependencies**

- `numpy`, `scipy.optimize.minimize` (SLSQP).

**Edge cases / failure modes**

- No 1 in BO/OW or multiple 1s → ValueError in `_validate_and_detect_indices`.
- BO/OW < 1 → ValueError.
- SLSQP fails or returns out-of-range → warnings + fallbacks and clipping; upper/lower can still be inconsistent if fallbacks used.
- `solve_lower_bounds` or `solve_upper_bounds` called before `solve_exact` → AssertionError.

---

## 3. interval_bwm.py

**Purpose:** Build interval weight matrices for all decision makers and criteria by running BWM for each (k, j) and storing lower/upper bounds.

**Role in flow:** Second (and third) step: called from `main.py` after load; produces IP consumed by `cloud_dm` and `visualizer`.

**Key class and method**

| Signature | Responsibility |
|-----------|----------------|
| `IntervalBWM_SciPy(BO, OW, K, C, A)` | Store BO, OW, K, C, A; IP = [None]*K. |
| `run() -> IP` | For each k, j run BWM_Solver_SciPy(BO[k][j], OW[k][j]).solve_all(); fill IP[k][j,i,0]=lower[i], IP[k][j,i,1]=upper[i]. Return list of (C,A,2) arrays. |

**Step-by-step logic**

```
1. for k in 0..K-1:
2.     interval_matrix = zeros((C, A, 2))
3.     for j in 0..C-1:
4.         solver = BWM_Solver_SciPy(BO[k][j], OW[k][j])
5.         result = solver.solve_all()
6.         for i in 0..A-1: interval_matrix[j,i,0]=result["lower"][i], [j,i,1]=result["upper"][i]
7.     IP[k] = interval_matrix
8. return IP
```

**Inputs**

- **Args:** `BO`, `OW` (from data_loader), `K`, `C`, `A` (int).
- **Config:** None.

**Outputs / side effects**

- **Returns:** `IP` — list of K arrays shape (C, A, 2). No files; mutates self.IP.

**Dependencies**

- `bwm_solver.BWM_Solver_SciPy`, `numpy`.

**Edge cases / failure modes**

- Any BWM_Solver_SciPy validation or solver failure for a (k,j) propagates and aborts run().
- If lower/upper were ever swapped in solver output, interval_bwm stores them as-is (visualizer later swaps for display if lower > upper).

---

## 4. cloud_dm.py

**Purpose:** Turn interval weight matrices IP into one aggregated cloud decision matrix (Ex, En, He) per (alternative, criterion).

**Role in flow:** Fourth step: called from `main.py` after interval_bwm; output feeds BiLevelOptimizer, Prioritization, Visualizer.

**Key class and methods**

| Signature | Responsibility |
|-----------|----------------|
| `CloudDecisionMatrix(IP, K, C, A)` | Store IP, K, C, A; cloud_matrix = None. |
| `_interval_to_cloud_params(lower, upper) -> (x_bar, sigma)` | x_bar = (lower+upper)/2, sigma = (upper-lower)/6. |
| `construct() -> cloud_matrix` | For each (k,j,i) compute x_bar, sigma; then for each (i,j) aggregate over k: Ex = mean(x_bar), En = mean(sigma) + sqrt(mean((x_bar-Ex)^2)), He = sqrt(mean((sigma-En)^2)). Return (A,C,3). |
| `get_cloud_matrix()` | Return self.cloud_matrix; raise if construct() not called. |
| `display()` | Print cloud matrix to stdout. |

**Step-by-step logic (construct)**

```
1. x_bar, sigma = zeros((K,C,A)), zeros((K,C,A))
2. for k,j,i: x_bar[k,j,i], sigma[k,j,i] = _interval_to_cloud_params(IP[k][j,i,0], IP[k][j,i,1])
3. cloud_matrix = zeros((A,C,3))
4. for i,j:
   - x_bars = x_bar[:,j,i], sigmas = sigma[:,j,i]
   - Ex = mean(x_bars), En = mean(sigmas) + sqrt(mean((x_bars-Ex)^2)), He = sqrt(mean((sigmas-En)^2))
   - cloud_matrix[i,j,:] = (Ex, En, He)
5. self.cloud_matrix = cloud_matrix; return cloud_matrix
```

**Inputs**

- **Args:** `IP` (list of (C,A,2) arrays), `K`, `C`, `A`.
- **Config:** None.

**Outputs / side effects**

- **Returns:** `cloud_matrix` ndarray (A, C, 3). No files. `display()` prints to stdout.

**Dependencies**

- `numpy` only.

**Edge cases / failure modes**

- `get_cloud_matrix()` or `display()` before `construct()` → ValueError.
- If upper < lower for some cell, sigma can be negative (implementation uses (upper-lower)/6; unclear if callers guarantee upper ≥ lower).

---

## 5. weightcal_bilevel_ooptimization.py

**Purpose:** Compute criteria weights by minimizing the maximum weighted hyper-entropy deviation (bi-level formulation) subject to simplex constraints.

**Role in flow:** Fifth step: called from `main.py` after cloud_dm; criteria weights passed to Prioritization.

**Key class and methods**

| Signature | Responsibility |
|-----------|----------------|
| `BiLevelOptimizer(cloud_matrix, A, C)` | Extract He = cloud_matrix[:,:,2]; store A, C; weights = None. |
| `solve() -> weights` | Minimize t over x=[W_1..W_C, t] s.t. t ≥ ±(He_ij*W_j - He_i0j0*W_j0) for all (i,j),(i0,j0), sum(W)=1, W≥0, t≥0. SLSQP. Normalize W; set self.weights, self.optimal_value. Return weights. |
| `get_weights()` | Return self.weights; raise if solve() not called. |
| `display()` | Print weights and He stats to stdout. |

**Step-by-step logic (solve)**

```
1. objective(x) = x[-1]   # t
2. ineq: for every (i,j), (i0,j0): t - (He_ij*W_j - He_i0j0*W_j0) ≥ 0, t + (He_ij*W_j - He_i0j0*W_j0) ≥ 0
3. eq: sum(W) - 1 = 0
4. bounds: W_j ≥ 0, t ≥ 0
5. x0 = [1/C,...,1/C, 0.1]; minimize with SLSQP
6. self.weights = clip(result.x[:-1], 0, None); normalize to sum 1; self.optimal_value = result.x[-1]; return self.weights
```

**Inputs**

- **Args:** `cloud_matrix` (A,C,3), `A`, `C`.
- **Config:** None. Solver options hardcoded.

**Outputs / side effects**

- **Returns:** 1D array of length C. `display()` prints to stdout. No files.

**Dependencies**

- `numpy`, `scipy.optimize.minimize`.

**Edge cases / failure modes**

- SLSQP may not converge; no check on result.success; weights are still extracted and normalized.
- Large A,C → many constraints (A*C*A*C*2); can be slow or memory-heavy.

---

## 6. prioritization.py

**Purpose:** Build weighted cloud matrix, compute CPIS/CNIS, distances to ideals, and ranking scores; rank alternatives.

**Role in flow:** Sixth step: called from `main.py` after BiLevelOptimizer; outputs used for display and by main’s export_to_excel.

**Key class and methods**

| Signature | Responsibility |
|-----------|----------------|
| `Prioritization(cloud_matrix, criteria_weights, A, C)` | Store inputs; weighted_matrix, CPIS, CNIS, d_plus, d_minus, ranking_scores, ranking_order = None. |
| `_compare_clouds(cloud1, cloud2) -> -1|0|1` | Definition 3: interval (Ex±3En), S = 2(ā-b_bar)-(ā-a_ul+b_bar-b_ul); if S>0 then 1, S<0 then -1; else tie-break by En, then He. |
| `_cloud_distance(cloud1, cloud2) -> float` | Definition 4: sqrt((Ex1-Ex2)^2 + |En1-En2| + |He1-He2|). |
| `construct_weighted_matrix()` | weighted_matrix[i,j,:] = cloud_matrix[i,j,:] * criteria_weights[j]. Return weighted_matrix. |
| `define_ideal_solutions()` | For each j: CPIS[j] = weighted_matrix[i*,j,:] s.t. i* = argmax_i (compare); CNIS[j] = argmin_i. Return CPIS, CNIS. |
| `calculate_distances()` | d_plus[i] = sum_j d(weighted_matrix[i,j], CPIS[j]); d_minus[i] = sum_j d(weighted_matrix[i,j], CNIS[j]). Return d_plus, d_minus. |
| `calculate_ranking_scores()` | RS[i] = d_minus[i]/(d_minus[i]+d_plus[i]); ranking_order = argsort(RS)[::-1]. Return ranking_scores, ranking_order. |
| `run() -> (ranking_scores, ranking_order)` | Call construct_weighted_matrix, define_ideal_solutions, calculate_distances, calculate_ranking_scores. Return last two. |
| `display_results()` | Print ranking scores, order, CPIS/CNIS, distances; raise if run() not called. |

**Step-by-step logic (run)**

```
1. construct_weighted_matrix()  # Ŷ_ij = Ỹ_ij * w_j
2. define_ideal_solutions()    # CPIS[j], CNIS[j] by _compare_clouds
3. calculate_distances()       # d_i^+, d_i^-
4. calculate_ranking_scores()  # RS_i, ranking_order
5. return ranking_scores, ranking_order
```

**Inputs**

- **Args:** `cloud_matrix` (A,C,3), `criteria_weights` (C,), `A`, `C`.
- **Config:** None.

**Outputs / side effects**

- **Returns:** From `run()`: (ranking_scores, ranking_order). Instance attributes set. `display_results()` prints to stdout. No files.

**Dependencies**

- `numpy` only.

**Edge cases / failure modes**

- `display_results()` before `run()` → ValueError.
- If d_minus[i]+d_plus[i] ≈ 0, denominator guarded by 1e-10; RS_i set to 0.
- CPIS/CNIS use pairwise comparison; ties broken by En then He; deterministic but order of equal clouds not specified by doc.

---

## 7. visualizer.py

**Purpose:** Produce three figure types: interval weights (error bars), Gaussian membership functions (x̄, σ from intervals), aggregated cloud droplets (Algorithm 1).

**Role in flow:** Seventh step: called from `main.py` after prioritization; optional file writes.

**Key class and methods**

| Signature | Responsibility |
|-----------|----------------|
| `BWMVisualizer(IP=None, cloud_matrix=None, K=None, C=None, A=None)` | Store optional IP, cloud_matrix, K, C, A. |
| `plot_interval_weights(save_path=None)` | Subplots K×C; each: errorbar(alternatives, mean, yerr=half-width) from IP[k][j,:,0/1]; ylim [0,1]. Save if save_path. plt.show(). |
| `plot_gaussian_membership_functions(save_path=None)` | Subplots C×A; each: for each k plot Gaussian from (x_bar, sigma) = ((L+U)/2, (U-L)/6) for IP[k][j,i]; x in [0,1]. Save if save_path; show. |
| `plot_cloud_droplets(n_drops=1000, save_path=None)` | Subplots C×A; each: sample n_drops via En'~N(En,He), x~N(Ex,En'); membership = exp(-(x-Ex)^2/(2*En'^2)); scatter(x, membership). ylim [0,1]. Save if save_path; show. |
| `plot_all(save_prefix=None)` | If IP: plot_interval_weights(prefix+_interval_weights.png), plot_gaussian_membership_functions(prefix+_gmf.png). If cloud_matrix: plot_cloud_droplets(500, prefix+_cloud_droplets.png). |

**Step-by-step logic (plot_interval_weights)**

```
1. if IP is None: raise ValueError
2. fig, axes = subplots(K, C)
3. for k,j: means = (lower+upper)/2, errors = |upper-lower|/2; if lower>upper swap; errorbar(1..A, means, yerr=errors); ylim [0,1]
4. if save_path: savefig(save_path, dpi=300, bbox_inches='tight'); show()
```

**Inputs**

- **Args:** Constructor: optional IP, cloud_matrix, K, C, A. Plot methods: optional save_path; plot_cloud_droplets: n_drops.
- **Config:** None. `plot_all` uses n_drops=500 and save_prefix from main.

**Outputs / side effects**

- **Files:** If save_path/save_prefix given: PNGs (e.g. bwm_results_interval_weights.png, _gmf.png, _cloud_droplets.png). dpi=300.
- **Display:** plt.show() for each figure (blocks or displays per backend).
- **Console:** plot_all prints "Generating visualizations...", "1/3: ...", etc.

**Dependencies**

- `numpy`, `matplotlib.pyplot`, `matplotlib.gridspec` (GridSpec imported but usage not verified in snippets).

**Edge cases / failure modes**

- Missing IP or cloud_matrix when required by a plot method → ValueError.
- K or C or A wrong vs actual data → shape/index errors.
- plot_cloud_droplets: He or En_prime ≤ 0 guarded by 1e-6; En_prime negative after normal can still occur (then max(En_prime, 1e-6) used).

---

## 8. main.py

**Purpose:** Orchestrate full pipeline (load → interval BWM → cloud matrix → criteria weights → prioritization → visualization → Excel export) and print progress/summary.

**Role in flow:** Entry point. Calls all other core modules in sequence.

**Key functions**

| Signature | Responsibility |
|-----------|----------------|
| `main()` | Run steps 1–8: load Excel, run IntervalBWM, recompute exact weights for display, build CloudDecisionMatrix, BiLevelOptimizer.solve(), Prioritization.run(), BWMVisualizer.plot_all(), export_to_excel(). On exception in a step, print and return. |
| `export_to_excel(exact_weights, IP, cloud_matrix, weighted_matrix, CPIS, CNIS, d_plus, d_minus, ranking_scores, ranking_order, K, C, A)` | Write results.xlsx with multiple sheets (Dk_Optimal_Weights, Dk_Interval_Weights, Dk_Estimations, Dk_Uncertainty, Cloud_Decision_Matrix, Weighted_Cloud_Decision_Matrix, Ideal_Solutions, Ranking_Alternatives). |

**Step-by-step logic (main)**

```
1. excel_file = "Data_G5.xlsx"   # config: input file
2. BO, OW, K, C, A = load_bwm_data_from_excel(excel_file); on error return
3. IBWM = IntervalBWM_SciPy(BO, OW, K, C, A); IP = IBWM.run()
4. For display: exact_weights[k][j] = BWM_Solver_SciPy(BO[k][j], OW[k][j]).solve_all()["weights"]; print lower/main/upper
5. cdm = CloudDecisionMatrix(IP, K, C, A); cloud_matrix = cdm.construct(); print summary
6. optimizer = BiLevelOptimizer(cloud_matrix, A, C); criteria_weights = optimizer.solve(); optimizer.display()
7. prioritizer = Prioritization(cloud_matrix, criteria_weights, A, C); ranking_scores, ranking_order = prioritizer.run(); prioritizer.display_results()
8. viz = BWMVisualizer(...); viz.plot_all(save_prefix="bwm_results")
9. Print final summary (ranking)
10. If exact_weights, IP, cloud_matrix, prioritizer all set: export_to_excel(...); else skip with message
```

**export_to_excel (high level)**

```
- output_file = "results.xlsx"; writer = ExcelWriter(output_file, engine='openpyxl')
- For k: write P_data (A×C) to D{k+1}_Optimal_Weights; write IP lower/upper columns to D{k+1}_Interval_Weights
- For k: write x_bar (A×C) to D{k+1}_Estimations; write sigma (A×C) to D{k+1}_Uncertainty
- Write cloud (Ex,En,He) strings to Cloud_Decision_Matrix; weighted (Ex,En,He) to Weighted_Cloud_Decision_Matrix
- Write CPIS/CNIS rows to Ideal_Solutions; write ranking table to Ranking_Alternatives (sorted by Rank)
- writer.close()
```

**Inputs**

- **Config (hardcoded):** `excel_file` (e.g. "Data_G5.xlsx"), `output_file` "results.xlsx" inside export_to_excel, `save_prefix` "bwm_results".
- **Files read:** One Excel file path set in main.

**Outputs / side effects**

- **Files:** `results.xlsx` (if export runs); `bwm_results_interval_weights.png`, `bwm_results_gmf.png`, `bwm_results_cloud_droplets.png` (from visualizer).
- **Console:** Extensive print (steps, weights table, cloud summary, optimizer display, prioritization display, final ranking, export status). On error: print and traceback for export.

**Dependencies**

- `numpy`, `pandas`, `data_loader`, `interval_bwm`, `cloud_dm`, `weightcal_bilevel_ooptimization`, `prioritization`, `visualizer`, `bwm_solver` (for exact_weights).

**Edge cases / failure modes**

- Any step exception: main prints and returns; later steps (e.g. export) skipped. If prioritizer fails, prioritizer set to None and export skipped.
- export_to_excel assumes all arrays match K, C, A; wrong lengths can cause index errors.
- Excel sheet names must be valid; long K may hit sheet name length limits. Unknown: openpyxl sheet name length limit handling.

---

## 9. requirements.txt

**Purpose:** Declare Python package dependencies for the project.

**Role in flow:** Used by humans/tooling (e.g. `pip install -r requirements.txt`); not imported by code.

**Content (summary)**

- numpy>=1.21.0  
- scipy>=1.7.0  
- pandas>=1.3.0  
- matplotlib>=3.4.0  
- openpyxl>=3.0.0  

**Inputs / outputs**

- No inputs at runtime. No side effects; not executed.

---

## High-level call graph (module → module)

```
main.py
  ├── data_loader.load_bwm_data_from_excel
  ├── interval_bwm.IntervalBWM_SciPy.run
  │     └── bwm_solver.BWM_Solver_SciPy.solve_all  (per k,j)
  ├── bwm_solver.BWM_Solver_SciPy.solve_all        (per k,j, for exact_weights display)
  ├── cloud_dm.CloudDecisionMatrix.construct
  ├── weightcal_bilevel_ooptimization.BiLevelOptimizer.solve
  ├── prioritization.Prioritization.run
  ├── visualizer.BWMVisualizer.plot_all
  └── main.export_to_excel
```

- **data_loader** → no other project modules  
- **bwm_solver** → no other project modules  
- **interval_bwm** → bwm_solver  
- **cloud_dm** → no other project modules  
- **weightcal_bilevel_ooptimization** → no other project modules  
- **prioritization** → no other project modules  
- **visualizer** → no other project modules  
- **main** → data_loader, interval_bwm, bwm_solver, cloud_dm, weightcal_bilevel_ooptimization, prioritization, visualizer  

---

## Critical execution paths (file sequence)

**Full pipeline (normal run):**

1. `main.py` (main)
2. `data_loader.py` (load_bwm_data_from_excel)
3. `interval_bwm.py` (IntervalBWM_SciPy.run) → repeatedly `bwm_solver.py` (BWM_Solver_SciPy.solve_all)
4. `main.py` (recompute exact_weights via bwm_solver for display)
5. `cloud_dm.py` (CloudDecisionMatrix.construct)
6. `weightcal_bilevel_ooptimization.py` (BiLevelOptimizer.solve)
7. `prioritization.py` (Prioritization.run)
8. `visualizer.py` (BWMVisualizer.plot_all)
9. `main.py` (export_to_excel)

**Minimal path to ranking (no plots, no Excel):**

1. main.py → data_loader  
2. main.py → interval_bwm → bwm_solver  
3. main.py → cloud_dm  
4. main.py → weightcal_bilevel_ooptimization  
5. main.py → prioritization  

**Data flow (simplified):**

- Excel → BO, OW, K, C, A  
- BO, OW → IP (interval_bwm + bwm_solver)  
- IP → cloud_matrix (cloud_dm)  
- cloud_matrix → criteria_weights (weightcal_bilevel_ooptimization)  
- cloud_matrix + criteria_weights → weighted_matrix, CPIS, CNIS, d_plus, d_minus, ranking_scores, ranking_order (prioritization)  
- IP, cloud_matrix → figures (visualizer)  
- exact_weights, IP, cloud_matrix, weighted_matrix, CPIS, CNIS, d_plus, d_minus, ranking_* → results.xlsx (main.export_to_excel)  

---

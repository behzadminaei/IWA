# Project Documentation

## 1. What the Project Is

**BWM-Based Multi-Criteria Group Decision Making (MCGDM) with Cloud Models**

- Implements a full pipeline for **multi-criteria group decision making** using:
  - **Best–Worst Method (BWM)** for interval weight derivation
  - **Cloud models** (Ex, En, He) to aggregate multiple decision makers’ interval opinions
  - **Bi-level optimization** to weight criteria by agreement (hyper-entropy)
  - **Cloud-based prioritization** (CPIS/CNIS, distances, ranking scores) to rank alternatives
- Input: Excel file with Best-to-Others (BO) and Others-to-Worst (OW) comparisons per decision maker and criterion.
- Output: Interval weights, cloud decision matrix, criteria weights, alternative ranking, Excel report, and optional plots.

---

## 2. Repository Structure (Core Code Only)

All core code lives in the **repository root**; there are no subfolders for application logic.

| File | Purpose |
|------|--------|
| `main.py` | Entry point; runs full pipeline and Excel export |
| `data_loader.py` | Load BO/OW data from Excel |
| `bwm_solver.py` | Single-criterion BWM: exact weights, lower/upper bounds |
| `interval_bwm.py` | Multi-DM interval BWM; builds interval weight matrices |
| `cloud_dm.py` | Build cloud decision matrix (Ex, En, He) from intervals |
| `weightcal_bilevel_ooptimization.py` | Bi-level optimization for criteria weights |
| `prioritization.py` | Weighted cloud matrix, CPIS/CNIS, distances, ranking scores |
| `visualizer.py` | Plots: interval weights, GMF, cloud droplets |
| `requirements.txt` | Python dependencies |

**Excluded from this documentation (as requested):** example/demo scripts, notebooks, result files (e.g. `results*.xlsx`, `bwm_results_*.png`), `__pycache__`, `venv`, and other generated or experimental artifacts.

---

## 3. End-to-End Flow

```
Excel input (BO, OW)
        │
        ▼
┌───────────────────┐
│ 1. data_loader    │  load_bwm_data_from_excel() → BO, OW, K, C, A
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 2. bwm_solver     │  (used inside interval_bwm) exact + lower/upper weights
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 3. interval_bwm   │  IntervalBWM_SciPy.run() → IP (interval weight matrices)
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 4. cloud_dm       │  CloudDecisionMatrix.construct() → cloud_matrix (A×C×3)
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 5. weightcal_     │  BiLevelOptimizer.solve() → criteria_weights (C,)
│    bilevel_        │
│    optimization   │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 6. prioritization │  Prioritization.run() → ranking_scores, ranking_order
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 7. visualizer     │  BWMVisualizer.plot_all() → interval weights, GMF, droplets
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 8. main.py        │  export_to_excel() → results.xlsx (multi-sheet)
└───────────────────┘
```

- **K** = number of decision makers  
- **C** = number of criteria  
- **A** = number of alternatives  
- **IP** = list of interval matrices; `IP[k]` shape (C, A, 2)  
- **cloud_matrix** shape (A, C, 3): [:, :, 0]=Ex, [:, :, 1]=En, [:, :, 2]=He  

---

## 4. Module Responsibilities

| Module | Main class/function | Responsibility |
|--------|---------------------|----------------|
| **data_loader** | `load_bwm_data_from_excel(filepath)` | Parse Excel: row 1 = criteria headers, row 2 = alternative labels, then BO/OW rows per DM. Returns BO, OW, K, C, A. Structure assumed: 2 header rows, then K pairs of BO/OW rows. |
| **bwm_solver** | `BWM_Solver_SciPy` | Single BWM problem: input BO, OW vectors; validates (single 1 in BO and OW); solves exact + lower/upper via SLSQP; returns weights, e_star, lower, upper. |
| **interval_bwm** | `IntervalBWM_SciPy` | For each DM k and criterion j, runs BWM_Solver_SciPy on BO[k][j], OW[k][j]; fills IP[k] with lower ([:,:,0]) and upper ([:,:,1]) bounds. |
| **cloud_dm** | `CloudDecisionMatrix` | From IP: x̄ = (L+U)/2, σ = (U−L)/6; aggregates over DMs: Ex = mean(x̄), En = mean(σ)+√(variance of x̄), He = √(mean((σ−En)²)). Output (A, C, 3). |
| **weightcal_bilevel_ooptimization** | `BiLevelOptimizer` | Min-max over |He_ij·W_j − He_{i0,j0}·W_{j0}| s.t. ΣW_j=1, W_j≥0. Uses SciPy SLSQP; returns optimal criteria weights. |
| **prioritization** | `Prioritization` | (1) Weighted cloud matrix Ŷ_ij = Ỹ_ij×w_j. (2) CPIS/CNIS per criterion via cloud comparison (Definition 3). (3) Distances via Definition 4; d_i^+, d_i^-. (4) RS_i = d_i^-/(d_i^-+d_i^+); rank descending. |
| **visualizer** | `BWMVisualizer` | Three plots: interval weights (error bars), Gaussian membership functions (x̄, σ), aggregated cloud droplets (Algorithm 1). Optional save paths. |
| **main** | `main()`, `export_to_excel(...)` | Runs steps 1–7 in order; prints weights (lower/main/upper); writes results.xlsx (optimal weights, interval weights, estimations, uncertainty, cloud matrix, weighted cloud, ideal solutions, ranking). |

---

## 5. Run, Test, Deploy

### Run

- **Full pipeline:**  
  Set the input file in `main.py` (variable `excel_file`, ~line 45), then:
  ```bash
  python main.py
  ```
- **Dependencies:**  
  `pip install -r requirements.txt`  
  (Uses: numpy, scipy, pandas, matplotlib, openpyxl.)

### Test

- No automated test suite or test runner is present in the repo.  
- **Unknown:** Whether tests exist elsewhere or are planned.

### Deploy

- No deployment or container setup is present.  
- Execution is local CLI; input = Excel path in code; output = console, `results.xlsx`, and optional PNGs from the visualizer.

---

## 6. Configuration

| What | Where | How used |
|------|--------|----------|
| **Input Excel file** | `main.py`, variable `excel_file` (e.g. `"Data_G5.xlsx"`) | Passed to `load_bwm_data_from_excel(excel_file)`. Change this string to switch datasets. |
| **Excel output path** | `main.py`, inside `export_to_excel()`: `output_file = "results.xlsx"` | All exported tables written to this single workbook (multiple sheets). |
| **Python dependencies** | `requirements.txt` | Package list for install; no in-code config. |
| **Visualization save prefix** | `main.py`: `viz.plot_all(save_prefix="bwm_results")` | Filenames like `bwm_results_interval_weights.png`, etc. |
| **Cloud droplet count** | `visualizer.py`: `plot_cloud_droplets(n_drops=500, ...)` (called from `plot_all`) | Number of droplets per subplot. |

No config files (e.g. YAML/JSON/env) are used; all of the above are hardcoded in the listed files.

---

## 7. Data Contract (Excel Input)

- **Expected layout:**  
  - Row 1: criteria headers (e.g. C1, C2, …).  
  - Row 2: alternative labels (e.g. A1, A2, …) repeated per criterion.  
  - Rows 3 onward: for each decision maker, two rows — first row type BO, second OW; columns 1–2 = DM ID and BO/OW label; then blocks of A columns per criterion.  
- **Inferred dimensions:** K from number of BO/OW row pairs; A from first repeat of alternative label in row 2; C from total criterion columns / A.

---

*Document generated from repository scan. Only production/core components are described; examples, demos, and generated outputs are excluded as specified.*

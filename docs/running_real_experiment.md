# Running the real experiment

The normal CI workflow validates installation, linting and unit tests.

The separate **Run real experiment** workflow executes the actual end-to-end predictive-maintenance experiment:

1. downloads the UCI AI4I dataset
2. generates EDA outputs
3. trains the baseline comparison
4. runs stratified cross-validation
5. runs calibration, threshold-cost, permutation-importance and robustness analysis
6. builds the final project summary
7. builds a compact Git-tracked evidence pack

## How to run it

In GitHub:

1. Open **Actions**
2. Select **Run real experiment**
3. Select the branch you want to run
4. Leave **Commit the generated evidence pack** enabled
5. Click **Run workflow**

The workflow uploads the full `results/` directory as a GitHub Actions artifact for 90 days.

It also commits a smaller `evidence/` directory containing selected tables, figures, `RESULTS.md` and `final_project_summary.json`.

Raw datasets and transient model outputs remain excluded from Git.

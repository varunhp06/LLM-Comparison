# LLM Comparison

A data analysis and machine learning project that explores, models, and clusters a dataset of large language models (LLMs) across performance, cost, and efficiency metrics — built in a single Jupyter/Colab notebook.

## What's in this repo

- **`LLM_Comparison.ipynb`** — the full analysis pipeline: data preprocessing, exploratory data analysis, benchmark-score prediction, and clustering
- **`llm_comparison_dataset.csv`** — the source dataset (200 rows, 70 unique models across 8 providers)

> **Note:** the dataset (model names, benchmark scores, pricing, etc.) appears to be synthetically generated for practice/coursework rather than scraped from real published benchmarks — model names like `DeepSeek-4` or `Llama-8` don't correspond to actual released versions. Treat the analysis as a methodology demo rather than a factual comparison of real LLMs.

## Dataset

`llm_comparison_dataset.csv` has 200 rows and 15 columns:

| Column | Description |
|---|---|
| `Model` | Model name (e.g. `Llama-8`, `DeepSeek-4`) |
| `Provider` | One of: Deepseek, Meta AI, AWS, Anthropic, Google, OpenAI, Cohere, Mistral AI |
| `Context Window` | Max context length (tokens) |
| `Speed (tokens/sec)` | Generation throughput |
| `Latency (sec)` | Response latency |
| `Benchmark (MMLU)` | MMLU benchmark score |
| `Benchmark (Chatbot Arena)` | Chatbot Arena score |
| `Open-Source` | 1 = open-source, 0 = closed-source |
| `Price / Million Tokens` | Cost per million tokens |
| `Training Dataset Size` | Size of training data |
| `Compute Power` | Relative compute power used |
| `Energy Efficiency` | Relative energy efficiency |
| `Quality Rating`, `Speed Rating`, `Price Rating` | 1–3 categorical ratings |

There are 70 unique model names with multiple rows each (likely repeated benchmark runs), which the notebook aggregates in its first step.

## Notebook Walkthrough

The notebook is a linear pipeline; each cell reads the previous cell's output CSV and writes a new one.

1. **Aggregation** — groups the raw data by `Model` (mean of numeric columns, most frequent value for categoricals) → `llm_comparison_aggregated.csv`
2. **Min-Max scaling** — scales numeric columns to [0, 1] → `llm_comparison_scaled.csv`
3. **Standardization** — z-score standardizes numeric columns (from the aggregated, *unscaled* data) → `llm_comparison_standardized.csv`
4. **Encoding** — extracts a numeric `Model_number` from each model name, maps `Provider` to a numeric code, and combines them into a compact `Model_encoded` identifier → `llm_comparison_efficient_encoded.csv`
5. **Benchmark prediction** — exploratory plots (distributions, correlation heatmap, scatter plots), feature engineering (efficiency ratio, price tier, one-hot encoding), then trains and compares Linear Regression, Ridge, Random Forest, and Gradient Boosting to predict `Benchmark (MMLU)`, selects the best performer, tunes it with `GridSearchCV`, and writes a summary to `llm_performance_insights.md`
6. **Clustering** — K-means and hierarchical clustering over all standardized features, elbow-method and silhouette analysis to pick *k*, PCA for 2D/3D visualization, radar charts of cluster profiles, and identification of the best model per cluster by MMLU, Chatbot Arena, quality, and efficiency

## Known Issue: file naming mismatch

The notebook can't currently be run top-to-bottom without a manual fix. Step 5 expects two files — `llm_comparison_scaled_encoded.csv` and `llm_comparison_standardized_encoded.csv` — but step 4 only encodes the *standardized* data and saves it as `llm_comparison_efficient_encoded.csv`. To run the notebook end-to-end you'll need to either:

- run the encoding cell (cell 4) a second time against `llm_comparison_scaled.csv` and save the result as `llm_comparison_scaled_encoded.csv`, and rename `llm_comparison_efficient_encoded.csv` to `llm_comparison_standardized_encoded.csv`, **or**
- adjust the file paths in cells 5 and 6 to match the actual output of cell 4.

## Getting Started

### Option 1: Google Colab

Open the notebook directly via the badge at the top of `LLM_Comparison.ipynb`, upload `llm_comparison_dataset.csv` to the Colab session, and run the cells in order (applying the fix above before cells 5–6).

### Option 2: Run locally

```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly scipy jupyter
jupyter notebook LLM_Comparison.ipynb
```

Make sure `llm_comparison_dataset.csv` is in the same directory as the notebook before running the first cell.

## Outputs

Running the notebook end-to-end produces, in the working directory:

- Intermediate CSVs: `llm_comparison_aggregated.csv`, `llm_comparison_scaled.csv`, `llm_comparison_standardized.csv`, `llm_comparison_efficient_encoded.csv`
- Static plots (`.png`): benchmark distributions, correlation matrices, boxplots by provider, scatter plots, feature importance, elbow curve, hierarchical dendrogram, PCA scatter, cluster comparisons
- Interactive plots (`.html`): 3D PCA cluster view, cluster profile radar charts (via Plotly)
- `llm_performance_insights.md` — an auto-generated summary of the best-performing model, top predictive features, provider rankings, and recommendations

## License

No license file is currently included in this repository.

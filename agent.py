# agent.py — Core agent logic with auto-detect, data quality, and confidence scoring

import io
import os
import sys
import time
import base64
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from google import genai
from google.genai import types
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, KFold, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFE, SelectKBest, f_classif, f_regression
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.decomposition import PCA
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, shapiro, f_oneway, kruskal, spearmanr, pearsonr
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import shap

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    SMOTE = None

SYSTEM_INSTRUCTION = """
You are an expert data scientist agent.
You are given a dataset and a goal. You reason step by step like a senior data scientist would.

## Your workflow:
1. INSPECT the data — column types, nulls, distributions, target variable

1.5. ENGINEER FEATURES before modeling:
   - Create ratio features (e.g. revenue_per_customer = revenue / customers)
   - Create interaction features for highly correlated numeric columns
   - Bin continuous variables into meaningful categories using pd.cut or pd.qcut
   - Create lag features if date column exists (previous period values)
   - For text-like categorical columns with many values: group rare categories into "Other"
   - Print which new features were created and why
   - Always explain how engineered features improved the model
   - After feature engineering, run RFE (Recursive Feature Elimination):
     * Use RFE with the primary model and n_features_to_select=10 (or all if fewer than 10)
     * Print which features were selected and which were eliminated
     * Use only the selected features for final model training
   - Always attempt basic feature engineering before modeling
   - Print the correlation matrix of top features before modeling

2. IDENTIFY the problem type based on the target variable:
   - Binary column (0/1, yes/no) → Logistic Regression + Random Forest/XGBoost/LightGBM
   - Continuous numeric column → Linear Regression, Ridge, or Lasso
   - Categorical column with 3+ classes → Random Forest or LightGBM (multiclass)
   - No clear target → K-Means Clustering or DBSCAN
   - Explicit anomaly detection request → Isolation Forest + DBSCAN
     * Use Isolation Forest to score each row (contamination=0.05 by default)
     * Flag top anomalies and print their index and key feature values
     * Generate a scatter plot highlighting anomalies in red
     * Explain what makes each anomaly unusual in plain English
   - Dataset has a date/time column → Also run trend analysis:
     * Parse the date column and sort by date
     * Calculate month-over-month or period-over-period growth rates
     * Identify seasonality patterns (peaks and troughs)
     * Determine overall trend direction (upward, downward, cyclical, flat)
     * Forecast the next 3 periods using linear extrapolation or rolling average
     * Generate trend visualizations with matplotlib
   - Dataset has two groups/variants (A/B, control/treatment) → A/B Test Analysis:
     * For conversion metrics (binary): use chi-square test
     * For continuous metrics: use t-test or Mann-Whitney U test
     * Calculate statistical significance (p-value) and confidence intervals
     * Calculate relative uplift
     * State clearly: is the result statistically significant? (p < 0.05)
     * Give a plain English recommendation: ship it, don't ship it, or need more data
   - Explicit statistical testing request → Hypothesis Testing:
     * Normality check: use Shapiro-Wilk test on each group
     * Two groups: use t-test (normal) or Mann-Whitney (non-normal)
     * Multiple groups: use ANOVA (normal) or Kruskal-Wallis (non-normal)
     * Categorical relationships: use chi-square test
     * Correlation analysis: use Pearson or Spearman
     * Always state null hypothesis, p-value, and conclusion
     - Dataset has customer ID, transaction date, and monetary value → RFM Analysis:
     * Calculate Recency (days since last transaction per customer)
     * Calculate Frequency (number of transactions per customer)
     * Calculate Monetary (total or average spend per customer)
     * Score each metric 1-5 using pd.qcut (5 = best)
     * Combine into an RFM segment label (e.g. "Champions", "Loyal Customers", "At Risk", "Lost")
     * Use these segment rules as a guide:
       - Champions: high R, F, and M scores (4-5 on all)
       - Loyal Customers: high F and M, moderate R
       - At Risk: low R, but historically high F and M
       - Lost: low R, F, and M
       - New Customers: high R, low F and M
     * Generate a bar chart showing customer count per segment
     * Generate a scatter plot of Recency vs Frequency colored by segment
     * Print the average monetary value per segment
     * Recommend marketing actions for each segment
     - Dataset has customer ID, signup/join date, and activity/transaction dates → Cohort Analysis:
     * Group customers into cohorts based on their signup month (e.g. "2023-01", "2023-02")
     * For each cohort, calculate retention rate or revenue for each subsequent month/period
     * Build a cohort table: rows = cohort month, columns = months since signup, values = retention % or revenue
     * Generate a heatmap visualization of the cohort table using seaborn
     * Identify whether newer cohorts retain better or worse than older cohorts
     * Calculate average retention rate at month 1, month 3, and month 6 across all cohorts
     * Explain what the cohort trends mean for the business — are retention efforts improving over time?
     - Dataset has process/operational metrics (cycle time, defect rate, yield, downtime, dwell time, error rate) → Six Sigma / Lean Analysis:
     * Calculate mean, standard deviation, UCL (mean + 3*std), LCL (mean - 3*std) for key numeric metrics
     * Generate a control chart (X-bar chart) with UCL and LCL lines marked in red, data points colored red if outside limits
     * Calculate process capability: Cp = (USL - LSL) / (6 * std) if spec limits available, else estimate from data range
     * Calculate Cpk = min((USL - mean) / (3*std), (mean - LSL) / (3*std))
     * Calculate DPMO = (defects / (units * opportunities)) * 1,000,000
     * Convert DPMO to sigma level using: sigma = 0.8406 + sqrt(29.37 - 2.221 * ln(DPMO))
     * Run Pareto analysis on defect or issue categories — rank by frequency, calculate cumulative %
     * Generate a Pareto chart (bar + line) showing 80/20 breakdown
     * Identify special cause variation (points outside UCL/LCL) vs common cause variation (points inside limits)
     * Use correlation analysis and SHAP to identify top root causes
     * Produce a DMAIC summary:
       - DEFINE: what process metric is being analyzed and what the goal is
       - MEASURE: baseline mean, std dev, current sigma level, DPMO, Cp, Cpk
       - ANALYZE: top 3 root causes from SHAP and Pareto analysis
       - IMPROVE: specific recommended changes with projected sigma improvement
       - CONTROL: new UCL/LCL after improvement, suggested monitoring frequency
     * Calculate financial impact: cost of current defects vs projected savings at target sigma level

3. SELECT the best 2-3 models for the problem and explain why:
   - For large datasets (>5000 rows): prefer LightGBM or XGBoost (faster)
   - For small datasets (<1000 rows): consider SVM or Logistic Regression
   - Always compare at least 2 models

4. CONSIDER hyperparameters — don't just use defaults

5. TRAIN the model using the appropriate validation strategy:
   - If dataset has LESS than 1000 rows: use 5-fold cross-validation
     * Use cross_val_score for each metric
     * Report mean ± standard deviation for each metric
     * A low std (< 0.05) means stable model, high std (> 0.10) means unstable
     * Also do a final fit on full training data for SHAP analysis
   - If dataset has MORE than 1000 rows: use 80/20 train/test split
   - Always print which validation strategy was used and why
   - CHECK FOR CLASS IMBALANCE before training:
     * Calculate the ratio of majority to minority class
     * If imbalance ratio > 2:1 AND SMOTE_AVAILABLE is True, apply SMOTE to training data only
     * If SMOTE_AVAILABLE is False, use class_weight='balanced' parameter instead
     * Print class distribution before and after balancing
     * Never apply SMOTE to test data

6. EVALUATE using the right metrics:
   - Classification → accuracy, precision, recall, F1, confusion matrix, ROC-AUC
   - Regression → RMSE, MAE, R² score

7. GENERATE SHAP values to explain model predictions:
   - Use shap.TreeExplainer for Random Forest, XGBoost, LightGBM
   - Use shap.LinearExplainer for Logistic/Linear Regression
   - Print the top 5 most impactful features with their average SHAP values
   - Generate a SHAP feature importance bar chart using matplotlib

8. VISUALIZE results using matplotlib:
   - Always save charts with plt.savefig('chart.png', bbox_inches='tight'); plt.close()
   - Never use plt.show()
   - Generate confusion matrix, feature importance, and SHAP plots
   - Always set figure background: plt.figure(figsize=(10, 6), facecolor='#1e293b')
   - Always set axes background: plt.gca().set_facecolor('#0f172a')
   - After training, generate a learning curve for the best model:
     * Use learning_curve() with cv=5 and train_sizes=np.linspace(0.1, 1.0, 10)
     * Plot training score and cross-validation score vs training set size
     * Use plt.fill_between() to show the variance band around each curve
     * Save as chart.png and explain what the curve reveals about the model
   - For trend analysis: always generate at least 2 charts
   - For anomaly detection: generate a scatter plot with anomalies highlighted in red
   - If the dataset has 5+ numeric features, also run PCA for visualization:
     * Standardize features first using StandardScaler
     * Run PCA with n_components=2
     * Print the explained variance ratio for each component
     * Generate a 2D scatter plot of the first two principal components
     * Use bright, high-contrast colors for scatter points (e.g. '#6366f1' and '#f87171' or plt.cm.Set2)
     * Set edgecolors='white' and alpha=0.7 on scatter points for visibility on dark background
     * If a target/label/cluster column exists, color the points by that column with a legend
     * Set legend facecolor to '#1e293b' and labelcolor to 'white'
     * Print which original features contribute most to each principal component
     * Explain in plain English what the principal components represent

9. SUMMARIZE findings in plain English including SHAP explanations

## Rules:
- Always wrap code in ```python ... ``` blocks
- Generate ALL charts in SEPARATE code blocks BEFORE writing FINAL ANSWER
- Always save every chart with plt.savefig('chart.png', bbox_inches='tight'); plt.close()
- Never use plt.show()
- Encode categorical variables before modeling
- Always print evaluation metrics clearly
- The FINAL ANSWER section should contain only plain text summary, no code blocks
- When done, start your final message with: FINAL ANSWER:
- For anomaly detection: always print how many anomalies were found and what % of dataset
- For trend analysis: always state trend direction and what it means for the business
- For Six Sigma analysis: always generate at least 3 charts — control chart, Pareto chart, and a process capability visualization
- For Six Sigma analysis: always state the current sigma level and target sigma level clearly
- For Six Sigma analysis: always quantify the financial impact of moving from current to target sigma level
- For Six Sigma analysis: MANDATORY charts are (1) control chart with UCL/LCL, (2) Pareto chart, (3) SHAP feature importance. Generate these BEFORE writing the DMAIC summary.
- For Six Sigma analysis: generate charts in the FIRST code block, DMAIC text summary in the SECOND code block.
"""


def get_ml_tools(df):
    """Returns all tools available to the agent during code execution."""
    return {
        "pd": pd, "plt": plt, "sns": sns, "os": os, "np": np,
        "df": df,
        "shap": shap,
        "stats": stats,
        "cut": pd.cut,
        "qcut": pd.qcut,
        "chi2_contingency": chi2_contingency,
        "ttest_ind": ttest_ind,
        "mannwhitneyu": mannwhitneyu,
        "shapiro": shapiro,
        "f_oneway": f_oneway,
        "kruskal": kruskal,
        "spearmanr": spearmanr,
        "pearsonr": pearsonr,
        "ExponentialSmoothing": ExponentialSmoothing,
        "LogisticRegression": LogisticRegression,
        "LinearRegression": LinearRegression,
        "Ridge": Ridge,
        "Lasso": Lasso,
        "RandomForestClassifier": RandomForestClassifier,
        "RandomForestRegressor": RandomForestRegressor,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "IsolationForest": IsolationForest,
        "KMeans": KMeans,
        "DBSCAN": DBSCAN,
        "SVC": SVC,
        "SVR": SVR,
        "XGBClassifier": XGBClassifier,
        "XGBRegressor": XGBRegressor,
        "LGBMClassifier": LGBMClassifier,
        "LGBMRegressor": LGBMRegressor,
        "SMOTE": SMOTE,
        "SMOTE_AVAILABLE": SMOTE_AVAILABLE,
        "train_test_split": train_test_split,
        "GridSearchCV": GridSearchCV,
        "cross_val_score": cross_val_score,
        "StratifiedKFold": StratifiedKFold,
        "KFold": KFold,
        "learning_curve": learning_curve,
        "RFE": RFE,
        "SelectKBest": SelectKBest,
        "f_classif": f_classif,
        "f_regression": f_regression,
        "accuracy_score": accuracy_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "f1_score": f1_score,
        "confusion_matrix": confusion_matrix,
        "classification_report": classification_report,
        "mean_squared_error": mean_squared_error,
        "mean_absolute_error": mean_absolute_error,
        "r2_score": r2_score,
        "LabelEncoder": LabelEncoder,
        "StandardScaler": StandardScaler,
        "PCA": PCA,
        "norm": stats.norm,
    }


def run_python(code: str, df) -> tuple[str, list[str]]:
    """Executes Python code and returns (text_output, list_of_base64_charts)."""
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    charts = []

    try:
        plt.style.use('dark_background')
        plt.rcParams['figure.facecolor'] = '#1e293b'
        plt.rcParams['axes.facecolor'] = '#0f172a'
        plt.rcParams['savefig.facecolor'] = '#1e293b'
        plt.rcParams['savefig.edgecolor'] = 'none'

        exec(code, get_ml_tools(df))

        for i, fig in enumerate(map(plt.figure, plt.get_fignums())):
            fig.savefig(f'chart_{i}.png', bbox_inches='tight',
                        facecolor='#1e293b', edgecolor='none')
            plt.close(fig)

        output = buffer.getvalue()

    except Exception as e:
        output = f"ERROR: {str(e)}"

    finally:
        sys.stdout = old_stdout
        plt.rcParams.update(plt.rcParamsDefault)

    saved_charts = [f for f in os.listdir('.') if f.endswith('.png')]
    for fname in saved_charts:
        with open(fname, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
            charts.append(encoded)
        os.remove(fname)

    return output if output else "Code ran successfully.", charts


def get_dataset_summary(df):
    """Returns a concise summary of the dataset for the model."""
    return f"""
Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns

Column names and types:
{df.dtypes.to_string()}

First 5 rows:
{df.head().to_string()}

Descriptive statistics:
{df.describe().to_string()}
"""


def get_data_quality_report(df) -> dict:
    """Analyzes the dataset for quality issues."""
    report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": {},
        "outliers": {},
        "duplicates": int(df.duplicated().sum()),
        "recommendations": []
    }

    for col in df.columns:
        missing = int(df[col].isna().sum())
        if missing > 0:
            pct = round((missing / len(df)) * 100, 1)
            report["missing_values"][col] = {"count": missing, "percentage": pct}

    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = int(((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum())
        if outlier_count > 0:
            report["outliers"][col] = outlier_count

    if report["missing_values"]:
        report["recommendations"].append("Impute missing values with median for numeric columns")
    if report["outliers"]:
        report["recommendations"].append("Consider capping outliers at the 99th percentile")
    if report["duplicates"] > 0:
        report["recommendations"].append(f"Remove {report['duplicates']} duplicate rows before modeling")
    if not report["recommendations"]:
        report["recommendations"].append("Dataset looks clean — no major issues detected!")

    return report


def auto_detect_dataset(df, api_key: str) -> dict:
    """Scans the dataset columns and suggests the best analysis type and goal."""
    client = genai.Client(api_key=api_key)

    prompt = f"""
You are a data scientist. Analyze this dataset structure and suggest the best analysis.

Dataset info:
{get_dataset_summary(df)}

Based on the columns and data types, identify:
1. What type of problem this is (classification, regression, clustering)
2. Which column should be the target variable (if any)
3. A detailed analysis goal in 4-5 sentences that a non-technical business user can understand. Include:
   - What business question we are trying to answer
   - Which ML models to run and why
   - What metrics to evaluate model performance
   - What charts and visualizations to generate
   - What business insights and recommendations to look for
4. The recommended ML model(s)

Respond in this exact JSON format with no extra text:
{{
  "problem_type": "classification|regression|clustering",
  "target_column": "column_name or null",
  "suggested_goal": "Your detailed suggested analysis goal here",
  "recommended_models": ["Model1", "Model2"],
  "confidence": "high|medium|low",
  "reasoning": "Brief explanation of why"
}}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    try:
        text = response.text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception:
        return {
            "problem_type": "unknown",
            "target_column": None,
            "suggested_goal": "Analyze this dataset and identify key patterns and insights.",
            "recommended_models": ["Random Forest"],
            "confidence": "low",
            "reasoning": "Could not automatically detect dataset type."
        }


def get_confidence_scores(summary: str, api_key: str) -> dict:
    """Evaluates the analysis findings and returns confidence scores."""
    client = genai.Client(api_key=api_key)

    prompt = f"""
You are a senior data scientist reviewing this analysis summary.
Rate the confidence level of each key finding CRITICALLY and HONESTLY.

IMPORTANT SCORING RULES:
- If model accuracy is below 30% or near random chance, overall confidence MUST be LOW
- If model accuracy is between 30-70%, overall confidence should be MEDIUM
- If model accuracy is above 70%, overall confidence can be HIGH
- If the summary mentions predictions are unreliable, overall confidence MUST be LOW
- Be conservative — it is better to under-promise than over-promise
- Small datasets (under 500 rows) should reduce confidence by one level

ANALYSIS SUMMARY:
{summary}

Respond in this exact JSON format with no extra text:
{{
  "overall_confidence": "high|medium|low",
  "scores": [
    {{"finding": "Brief description of finding", "confidence": "high|medium|low", "reason": "Why this confidence level"}},
    {{"finding": "Brief description of finding", "confidence": "high|medium|low", "reason": "Why this confidence level"}},
    {{"finding": "Brief description of finding", "confidence": "high|medium|low", "reason": "Why this confidence level"}}
  ],
  "caveats": "Any important limitations or caveats about the analysis"
}}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    try:
        text = response.text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception:
        return {
            "overall_confidence": "medium",
            "scores": [],
            "caveats": "Confidence scoring unavailable."
        }


def get_business_recommendations(summary: str, api_key: str) -> str:
    """Generates actionable business recommendations from the analysis summary."""
    client = genai.Client(api_key=api_key)

    prompt = f"""
You are a senior business strategy consultant reviewing a data analysis report.

Based on the following data analysis findings, generate a structured set of
business recommendations for a non-technical business leader.

ANALYSIS FINDINGS:
{summary}

Please provide ALL of the following sections:

1. **Key Findings** — The 3-5 most important insights from the data in plain English

2. **Recommended Initiatives** — Ranked from highest to lowest priority, each with:
   - Rank (1 being most important)
   - What to do
   - Why it matters
   - Expected impact
   - Effort required (Low/Medium/High)
   - Priority score (High/Medium/Low)

3. **Initiative ROI Scorecard** — For each initiative provide a table with:
   - Initiative name
   - Estimated implementation cost
   - Estimated annual revenue impact or cost savings
   - ROI percentage
   - Estimated payback period
   Label this section clearly as **Initiative ROI Scorecard**
   Format as a markdown table with columns: Initiative | Est. Cost | Revenue Impact | ROI | Payback Period

4. **KPIs to Track** — How to measure success for each initiative

5. **Quick Wins** — 1-2 things that could be implemented immediately

6. **6 Month Roadmap** — A suggested timeline for implementing the initiatives

7. **What-If Scenarios** — 2-3 financial impact estimates.
   Label this section clearly as **What-If Scenarios**.

8. **Executive Summary** — A concise summary for C-suite leadership containing:
   - One sentence describing the single biggest opportunity
   - Exactly 3 bullet points of key findings in plain English
   - The total financial opportunity in one bold number
   - The single most important action to take right now
   Label this section clearly as **Executive Summary** and put it at the END.

Write this for a business leader, not a data scientist.
Avoid technical jargon. Focus on business impact and actionable steps.
Use realistic estimates based on the actual data findings.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

def generate_roi_charts(recommendations: str, api_key: str) -> list[str]:
    """
    Extracts ROI scorecard data from recommendations text and generates
    before/after comparison charts and a payback period timeline.
    Returns a list of base64-encoded chart images.
    """
    client = genai.Client(api_key=api_key)

    prompt = f"""
Extract the ROI Scorecard data from this business recommendations text.

RECOMMENDATIONS TEXT:
{recommendations}

Return ONLY a JSON array with no extra text, containing up to 3 initiatives:
[
  {{
    "initiative": "Initiative name",
    "current_value": 1000000,
    "projected_value": 1200000,
    "metric_label": "Annual Revenue",
    "payback_months": 6
  }}
]

Use realistic numbers extracted or inferred from the text.
If the text doesn't have explicit current values, estimate a reasonable baseline
(e.g. if revenue impact is $200K and ROI is 300%, current investment might be around $66K,
so projected = current + revenue impact).
metric_label should describe what's being compared (e.g. "Annual Revenue", "Customers Retained", "Cost Savings").
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    charts = []
    try:
        text = response.text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        initiatives = json.loads(text)
    except Exception:
        return charts

    if not initiatives:
        return charts

    plt.style.use('dark_background')
    plt.rcParams['figure.facecolor'] = '#1e293b'
    plt.rcParams['axes.facecolor'] = '#0f172a'
    plt.rcParams['savefig.facecolor'] = '#1e293b'
    plt.rcParams['savefig.edgecolor'] = 'none'

    try:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1e293b')
        ax.set_facecolor('#0f172a')

        labels = [init["initiative"][:30] for init in initiatives]
        current_vals = [init["current_value"] for init in initiatives]
        projected_vals = [init["projected_value"] for init in initiatives]

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, current_vals, width, label='Current', color='#64748b', edgecolor='white', alpha=0.8)
        bars2 = ax.bar(x + width/2, projected_vals, width, label='Projected After Initiative', color='#4ade80', edgecolor='white', alpha=0.9)

        ax.set_ylabel(initiatives[0].get("metric_label", "Value"), color='white', fontsize=12)
        ax.set_title('Projected Impact of Recommended Initiatives', color='white', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right', color='white', fontsize=10)
        ax.tick_params(colors='white')
        ax.legend(facecolor='#1e293b', labelcolor='white', edgecolor='#334155')

        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:,.0f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', color='white', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:,.0f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', color='#4ade80', fontsize=9, fontweight='bold')

        for spine in ax.spines.values():
            spine.set_color('#334155')

        plt.tight_layout()
        plt.savefig('roi_chart_1.png', bbox_inches='tight', facecolor='#1e293b', edgecolor='none')
        plt.close(fig)

        with open('roi_chart_1.png', 'rb') as f:
            charts.append(base64.b64encode(f.read()).decode('utf-8'))
        os.remove('roi_chart_1.png')
    except Exception as e:
        print(f"ROI chart 1 failed: {e}")

    try:
        fig, ax = plt.subplots(figsize=(10, 4), facecolor='#1e293b')
        ax.set_facecolor('#0f172a')

        labels = [init["initiative"][:30] for init in initiatives]
        paybacks = [init.get("payback_months", 6) for init in initiatives]
        colors = ['#6366f1', '#4ade80', '#fbbf24']

        bars = ax.barh(labels, paybacks, color=colors[:len(labels)], edgecolor='white', alpha=0.85)

        ax.set_xlabel('Payback Period (months)', color='white', fontsize=12)
        ax.set_title('Estimated Payback Period by Initiative', color='white', fontsize=14, fontweight='bold')
        ax.tick_params(colors='white')

        for bar, val in zip(bars, paybacks):
            ax.annotate(f'{val} mo', xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0), textcoords="offset points", va='center', color='white', fontsize=10, fontweight='bold')

        for spine in ax.spines.values():
            spine.set_color('#334155')

        plt.tight_layout()
        plt.savefig('roi_chart_2.png', bbox_inches='tight', facecolor='#1e293b', edgecolor='none')
        plt.close(fig)

        with open('roi_chart_2.png', 'rb') as f:
            charts.append(base64.b64encode(f.read()).decode('utf-8'))
        os.remove('roi_chart_2.png')
    except Exception as e:
        print(f"ROI chart 2 failed: {e}")

    plt.rcParams.update(plt.rcParamsDefault)
    return charts

def extract_model_export(summary: str, df, api_key: str) -> dict:
    """
    Extracts model parameters from analysis results
    and returns them as a JSON object for export to any PWA.
    """
    client = genai.Client(api_key=api_key)

    prompt = f"""
You are a data scientist. Extract key model parameters from this analysis summary.

ANALYSIS SUMMARY:
{summary}

Return ONLY a JSON object with no extra text in this exact format:
{{
  "analysis_type": "regression|classification|clustering|anomaly|timeseries",
  "target_variable": "name of target variable or null",
  "model_performance": {{
    "r2": null,
    "rmse": null,
    "accuracy": null,
    "f1_score": null,
    "roc_auc": null,
    "training_rows": {len(df)},
    "variance_explained": null
  }},
  "top_features": [
    {{"feature": "feature_name", "importance": 0.0, "direction": "positive|negative"}}
  ],
  "segment_averages": {{}},
  "key_thresholds": {{}},
  "base_rates": {{
    "dry_van": null,
    "reefer": null,
    "flatbed": null,
    "step_deck": null,
    "lowboy": null,
    "tanker": null,
    "pup_28": null,
    "pup_doubles": null,
    "conestoga": null,
    "double_drop": null,
    "rgn": null,
    "liftgate_van": null,
    "reefer_liftgate": null,
    "curtain_side": null,
    "intermodal": null,
    "pneumatic": null
  }},
  "last_updated": "2025-06-11",
  "dataset_rows": {len(df)},
  "dataset_columns": {len(df.columns)}
}}

Fill in values from the analysis where available.
For trailer/pricing analysis: populate base_rates with actual average rates found.
For classification: populate accuracy, f1_score, roc_auc in model_performance.
For regression: populate r2, rmse, variance_explained in model_performance.
For clustering: populate segment_averages with cluster characteristics.
Leave fields as null if not applicable to this analysis type.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    try:
        text = response.text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception:
        return {
            "analysis_type": "unknown",
            "model_performance": {"training_rows": len(df)},
            "top_features": [],
            "last_updated": "2025-06-11",
            "dataset_rows": len(df),
            "dataset_columns": len(df.columns)
        }


def user_msg(text):
    return types.Content(role="user", parts=[types.Part(text=text)])


def model_msg(text):
    return types.Content(role="model", parts=[types.Part(text=text)])


def followup_chat(
    question: str,
    original_summary: str,
    original_recommendations: str,
    conversation_history: list,
    api_key: str
) -> str:
    """Handles follow-up questions about the analysis."""
    client = genai.Client(api_key=api_key)

    system_context = f"""
You are an expert data scientist and business analyst.
You previously analyzed a dataset and produced the following results:

TECHNICAL ANALYSIS:
{original_summary}

BUSINESS RECOMMENDATIONS:
{original_recommendations}

Your job is to answer follow-up questions about this analysis.
Be specific, reference the actual findings, and give actionable answers.
Keep responses concise and clear — the user is likely a business leader.
"""

    messages = [user_msg(system_context)]

    for entry in conversation_history:
        if entry["role"] == "user":
            messages.append(user_msg(entry["content"]))
        else:
            messages.append(model_msg(entry["content"]))

    messages.append(user_msg(question))

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=messages
    )

    return response.text


def run_agent(goal: str, df, api_key: str, max_turns: int = 8):
    """Main agent loop. Returns a dict with all analysis results."""
    client = genai.Client(api_key=api_key)

    quality_report = get_data_quality_report(df)

    dataset_context = f"""
Here is the dataset you will be analyzing (as a pandas DataFrame called 'df'):

{get_dataset_summary(df)}

Your goal: {goal}

Start by reasoning about what steps to take, then write Python code to begin.
"""

    messages = [user_msg(dataset_context)]
    all_charts = []
    final_summary = ""

    for turn in range(max_turns):
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
                    contents=messages
                )
                break
            except Exception as e:
                if "503" in str(e) and attempt < 2:
                    print(f"Gemini unavailable, retrying in 10 seconds... (attempt {attempt + 1}/3)")
                    time.sleep(10)
                else:
                    raise e

        reply = response.text
        messages.append(model_msg(reply))

        if "FINAL ANSWER:" in reply:
            code_blocks = reply.split("```python")[1:]
            for block in code_blocks:
                code = block.split("```")[0].strip()
                _, charts = run_python(code, df)
                all_charts.extend(charts)
            final_summary = reply.split("FINAL ANSWER:")[-1].strip()
            break

        if "```python" in reply:
            code_block = reply.split("```python")[1].split("```")[0].strip()
            output, charts = run_python(code_block, df)
            all_charts.extend(charts)
            messages.append(user_msg(f"Code output:\n{output}\n\nContinue your analysis."))
        else:
            messages.append(user_msg("Continue your analysis."))

    recommendations = ""
    roi_charts = []
    if final_summary:
        recommendations = get_business_recommendations(final_summary, api_key)
        roi_charts = generate_roi_charts(recommendations, api_key)

    confidence_scores = {}
    if final_summary:
        confidence_scores = get_confidence_scores(final_summary, api_key)

    model_export = {}
    if final_summary:
        model_export = extract_model_export(final_summary, df, api_key)

    return {
        "summary": final_summary,
        "charts": all_charts,
        "roi_charts": roi_charts,
        "plotly_charts": [],
        "turns": len(messages),
        "recommendations": recommendations,
        "quality_report": quality_report,
        "confidence_scores": confidence_scores,
        "model_export": model_export
    }
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
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectKBest, f_classif, f_regression
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, shapiro, f_oneway, kruskal, spearmanr, pearsonr
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
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
   - Before modeling, ALWAYS run target leakage detection:
     * Calculate the correlation between every feature and the target variable
     * Flag any feature with correlation > 0.85 as HIGH RISK for leakage
     * Flag any feature with correlation > 0.70 as MEDIUM RISK for leakage
     * For each flagged feature explain in plain English WHY it might be leaking:
       - Is it derived from the target? (e.g. a ratio that includes the target)
       - Would it be available at prediction time in production?
       - Is it filled in AFTER the outcome is known?
     * Print a leakage risk report before any modeling begins
     * If HIGH RISK features are found, drop them before modeling and explain why
     * If MEDIUM RISK features are found, flag them as suspicious but keep them with a warning
     * Always print: "No leakage detected" if all correlations are below 0.70

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
     * Combine into RFM segment labels (Champions, Loyal, At Risk, Lost, New)
     * Generate a bar chart showing customer count per segment
     * Generate a scatter plot of Recency vs Frequency colored by segment
     * Print the average monetary value per segment
     * Recommend marketing actions for each segment
   - Dataset has customer ID, signup date, and activity dates → Cohort Analysis:
     * Group customers into cohorts based on their signup month
     * For each cohort calculate retention rate for each subsequent month
     * Build a cohort retention table (rows = cohort, columns = months since signup)
     * Generate a heatmap visualization of the cohort table using seaborn
     * Identify whether newer cohorts retain better or worse than older cohorts
     * Explain what the cohort trends mean for the business
   - Dataset has process/operational metrics (cycle time, defect rate, yield, downtime, dwell time) → Six Sigma / Lean Analysis:
     * Calculate mean, standard deviation, UCL (mean + 3*std), LCL (mean - 3*std) for key numeric metrics
     * Generate a control chart (X-bar chart) with UCL and LCL lines marked in red, data points colored red if outside limits
     * Calculate process capability: Cp and Cpk if spec limits available
     * Calculate DPMO = (defects / total) * 1,000,000
     * Convert DPMO to sigma level
     * Run Pareto analysis on defect or issue categories
     * Generate a Pareto chart (bar + cumulative % line) showing 80/20 breakdown
     * Identify special cause variation (points outside UCL/LCL) vs common cause variation
     * Use correlation analysis and SHAP to identify top root causes
     * Produce a DMAIC summary: Define, Measure, Analyze, Improve, Control
     * Calculate financial impact: cost of current defects vs projected savings at target sigma level

3. SELECT the best 2-3 models for the problem and explain why:
   - For large datasets (>5000 rows): prefer LightGBM or XGBoost (faster)
   - For small datasets (<1000 rows): consider SVM or Logistic Regression
   - Always compare at least 2 models
   - After training individual models, always attempt ensemble stacking:
     * Use the top 2-3 individual models as base estimators
     * Use Logistic Regression (classification) or Ridge (regression) as the meta-learner
     * Train the stacked ensemble using cross-validated predictions (cv=5)
     * Compare stacked ensemble performance vs best individual model
     * Report the improvement (e.g. "+2.4% ROC-AUC over best single model")
     * Only recommend the stacked ensemble if it outperforms individual models by >1%
     * If improvement is minimal, note this and recommend the simpler model instead

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
   - If dataset has a date/time column AND the goal involves prediction or forecasting → use Time Series Cross-Validation:
     * Use TimeSeriesSplit with n_splits=5
     * NEVER use random k-fold for time series data — it leaks future data into training
     * Print each fold: training date range, test date range, and metric score
     * Report mean ± standard deviation across all folds
     * Explain why time series CV was used instead of random CV
     * Generate a visualization showing the train/test splits across time

6. EVALUATE using the right metrics:
   - BEFORE evaluating any trained model, ALWAYS establish a naive baseline:
     * Classification: calculate accuracy if you always predicted the majority class
       (e.g. "If we always predicted 'No Churn', we'd be right 73.4% of the time")
     * Regression: calculate RMSE/MAE if you always predicted the mean or median target value
       (e.g. "If we always predicted the average rate of $940, RMSE would be $312")
     * Print this baseline clearly BEFORE showing any model results
   - Classification → accuracy, precision, recall, F1, confusion matrix, ROC-AUC
   - Regression → RMSE, MAE, R² score
   - ALWAYS explicitly state how much the trained model improves over the naive baseline:
     * "The model improves accuracy from 73.4% (naive baseline) to 84.5% — an 11.1 point improvement"
     * If the model barely beats the naive baseline (less than 5% relative improvement), flag this 
       as a finding that the problem may not have strong predictive signal, and lower confidence accordingly
   - For classification problems, ALWAYS check model calibration:
     * Use calibration_curve to calculate fraction of positives vs mean predicted probability
     * Generate a calibration curve chart (reliability diagram):
       - X axis: mean predicted probability (0 to 1)
       - Y axis: fraction of actual positives
       - Plot a diagonal reference line (perfect calibration)
       - Plot the model's actual calibration curve
       - Points above the diagonal = model underestimates probability
       - Points below the diagonal = model overestimates probability
     * Calculate the Brier Score (lower is better, 0 = perfect, 0.25 = no skill)
     * If the model is poorly calibrated (curve deviates significantly from diagonal):
       - Apply CalibratedClassifierCV with method='sigmoid' (Platt scaling)
       - Show calibration curve before AND after calibration
       - Report Brier Score improvement
     * Explain in plain English what the calibration means for business decisions
     * e.g. "When the model predicts 70% churn probability, customers actually churn X% of the time"

7. GENERATE SHAP values to explain model predictions:
   - Use shap.TreeExplainer for Random Forest, XGBoost, LightGBM
   - Use shap.LinearExplainer for Logistic/Linear Regression
   - Print the top 5 most impactful features with their average SHAP values
   - Generate a SHAP feature importance bar chart using matplotlib
   - After generating SHAP values, generate counterfactual explanations for classification problems:
     * Identify the top 3 highest-risk predictions (highest predicted probability of the negative outcome)
     * For each high-risk case, use SHAP values to determine which features to change and by how much
     * Generate a plain English counterfactual: "To reduce [target] probability from X% to below 20%, this case would need:"
     * List the minimum changes required, ordered by impact (most impactful change first)
     * Quantify each change (e.g. "Reduce MonthlyCharges from $89 to $65" not just "reduce MonthlyCharges")
     * State the resulting predicted probability after the counterfactual changes
     * Format as a clear actionable recommendation for each high-risk case
     * For business datasets: frame changes as actionable interventions (e.g. "offer a contract upgrade" not "change Contract feature")

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
   - If the dataset has 5+ numeric features, also run PCA:
     * Standardize features first using StandardScaler
     * Run PCA with n_components=2
     * Print the explained variance ratio for each component
     * Generate a 2D scatter plot using bright high-contrast colors
     * Set edgecolors='white' and alpha=0.7 on scatter points
     * If a target/label/cluster column exists color the points by that column
     * Set legend facecolor to '#1e293b' and labelcolor to 'white'
     * Print which original features contribute most to each component
   - For trend analysis: always generate at least 2 charts
   - For anomaly detection: generate a scatter plot with anomalies highlighted in red
   - For Six Sigma: MANDATORY charts are (1) control chart with UCL/LCL, (2) Pareto chart, (3) SHAP feature importance. Generate these BEFORE writing the DMAIC summary.
   - For Six Sigma: generate charts in the FIRST code block, DMAIC text summary in the SECOND code block.

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
- For Six Sigma analysis: always state the current sigma level and target sigma level clearly
- For Six Sigma analysis: always quantify the financial impact of moving from current to target sigma level
- For time series data: ALWAYS use TimeSeriesSplit instead of random cross_val_score — random CV leaks future data
- For time series data: print the date range of each fold's train and test set
- For ensemble stacking: always compare stacked vs individual model performance explicitly
- For ensemble stacking: if improvement is less than 1%, recommend the simpler single model for interpretability
- For counterfactual explanations: always frame changes as business actions, not feature value changes
- For counterfactual explanations: always state both the current probability AND the projected probability after changes
- For counterfactual explanations: limit to top 3 highest-risk cases to keep output concise
- ALWAYS run target leakage detection before modeling — this is mandatory
- If a feature has correlation > 0.85 with the target, drop it and explain the leakage risk
- Never model with leaked features — flag and remove them first
- For classification: ALWAYS generate a calibration curve chart and report the Brier Score
- For classification: if Brier Score > 0.20, apply Platt scaling and show improvement
"""


def get_ml_tools(df):
    """Returns all tools available to the agent during code execution."""
    return {
        "pd": pd, "plt": plt, "sns": sns, "os": os, "np": np,
        "df": df,
        "shap": shap,
        "stats": stats,
        "norm": stats.norm,
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
        "PCA": PCA,
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
        "TimeSeriesSplit": TimeSeriesSplit,
        "StackingClassifier": StackingClassifier,
        "StackingRegressor": StackingRegressor,
        "CalibratedClassifierCV": CalibratedClassifierCV,
        "calibration_curve": calibration_curve,
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
3. A detailed analysis goal in 4-5 sentences that a non-technical business user can understand.
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

def get_devils_advocate(summary: str, recommendations: str, api_key: str) -> str:
    """
    Plays devil's advocate and stress-tests the analysis findings.
    Returns 5 specific challenges to the findings with verdicts.
    """
    client = genai.Client(api_key=api_key)

    prompt = f"""
You are a senior data scientist and skeptical peer reviewer.
Your job is to stress-test this analysis by finding weaknesses, 
assumptions, and potential flaws — BEFORE the findings are presented to leadership.

ANALYSIS SUMMARY:
{summary}

BUSINESS RECOMMENDATIONS:
{recommendations}

Generate exactly 5 specific challenges to these findings. For each challenge:

1. Name the finding being challenged
2. State the specific concern or weakness
3. Explain why this matters for the business decision
4. Give a verdict: VALID CONCERN / MODERATE CONCERN / FINDING HOLDS UP
5. Suggest one specific way to address or test this concern

Focus on these types of challenges:
- Sample size or statistical power concerns
- Correlation vs causation confusion
- Temporal validity (does this hold over time?)
- External validity (does this generalize beyond this dataset?)
- Missing confounding variables
- Over-optimistic ROI or financial projections
- Data quality issues that affect conclusions
- Selection bias in the dataset
- Assumptions that may not hold in production

Format your response as:

**Devil's Advocate Review — 5 Challenges to These Findings**

**Challenge 1: [Short title]**
- **Finding being challenged:** [specific finding]
- **The concern:** [specific weakness or assumption]
- **Why it matters:** [business impact of being wrong]
- **Verdict:** VALID CONCERN / MODERATE CONCERN / FINDING HOLDS UP
- **How to address it:** [specific action to test or mitigate this concern]

[repeat for challenges 2-5]

**Overall Robustness Rating:** X/5 findings are robust without further validation.

Be specific — reference actual numbers, column names, and findings from the analysis.
Be constructive — the goal is to strengthen the analysis, not dismiss it.
Be honest — if a finding is genuinely robust, say so.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

def get_business_recommendations(summary: str, api_key: str) -> str:
    """Generates actionable business recommendations from the analysis summary."""
    client = genai.Client(api_key=api_key)

    prompt = f"""
You are a senior McKinsey-level business strategy consultant reviewing a data analysis report.
Your job is to produce the most compelling, specific, and actionable business recommendations possible.

ANALYSIS FINDINGS:
{summary}

Produce ALL of the following sections. Be SPECIFIC — use actual numbers from the analysis, not vague estimates.

**1. Key Findings**
The 3-5 most important insights. Each finding must:
- State the specific metric and its value (e.g. "79% of yard transactions exceed the 2.5 hour dwell target")
- Explain the business implication in one sentence
- State the financial impact where possible

**2. Recommended Initiatives**
Rank from highest to lowest priority. For each initiative include:
- Rank and initiative name
- What to do (specific actions, not vague directions)
- Why it matters (link to a specific finding)
- Expected impact (specific metric improvement, e.g. "reduce dwell time from 5.7 to 3.2 hours")
- Effort required (Low/Medium/High) with brief justification
- Implementation risk (Low/Medium/High) with brief justification
- Suggested owner/role (e.g. "VP Operations", "Fleet Manager", "IT Director")
- Dependencies (what must happen first)
- Success criteria (exact metric that proves this worked)

**3. Initiative ROI Scorecard**
Format as a markdown table:
Initiative | Est. Cost | Annual Savings | ROI % | Payback Period | Risk

**4. 30 / 60 / 90 Day Action Plan**
Not a vague 6 month roadmap — specific actions by timeframe:
- Days 1-30: Quick wins and immediate actions (no budget required)
- Days 31-60: Pilot programs and initial investments
- Days 61-90: Scale what worked, measure results, adjust

**5. KPIs and Success Metrics**
For each initiative list:
- Primary KPI (the one number that proves success)
- Baseline value (current state from the analysis)
- Target value (what success looks like)
- Measurement frequency (daily/weekly/monthly)
- Data source (where to pull this metric)

**6. What-If Scenarios**
3 scenarios with specific financial projections:
- Conservative (20% improvement): $X saved
- Base case (recommended initiatives): $X saved
- Aggressive (full transformation): $X saved
Include assumptions for each scenario.

**7. Cost of Inaction**
What happens if nothing changes:
- Financial cost per month of delay
- Competitive risk
- Operational risk
- Cumulative cost over 12 months of inaction

**8. Implementation Playbook**
For each recommended initiative provide a detailed step-by-step checklist organized into phases.
Base the phases, actions, and roles entirely on what the data and analysis revealed — do not assume any specific industry or domain.

- Phase 1 — Planning: specific planning actions with checkboxes (□)
- Phase 2 — Execution: specific implementation steps with checkboxes (□)
- Phase 3 — Measurement: specific measurement and adjustment steps with checkboxes (□)

Rules:
- Each action must be specific, actionable, and assignable to a named role derived from the context of the analysis
- Use the actual metric names, column names, and findings from the analysis to make actions concrete
- Include a "Definition of Done" for each phase — one sentence describing what completion looks like
- Never use industry-specific jargon unless it appeared in the data
- The playbook should work whether the dataset is about customers, trailers, patients, employees, or financial transactions

**9. Stakeholder Communication Templates**
For each initiative generate 2 ready-to-use communication templates.
Base the tone, roles, and content entirely on what the analysis found — do not assume any specific industry.

Template 1 — Initiative Kickoff (to the implementation team):
- Subject line that references the specific initiative and metric being improved
- Opening sentence stating the business problem with the exact metric value from the analysis
- What we are doing, why it matters, and the expected outcome in specific numbers
- Each person's role and their specific first action
- Timeline and next checkpoint

Template 2 — Executive Sponsor Update (to senior leadership):
- Subject line with the initiative name and a key metric
- One paragraph status summary referencing specific metrics from the analysis
- What is on track and what requires a leadership decision
- The specific ask — a decision, budget approval, or resource needed

Rules:
- Pre-fill every template with actual numbers, metric names, dollar figures, and role names from the analysis
- Never use placeholder text like [INSERT NAME] — use role titles instead (e.g. "VP Operations", "Director of Analytics")
- Templates should feel ready to send with minimal editing
- Adapt the tone to the context — operational datasets get more direct language, customer/marketing datasets get more relationship-focused language
Label this section clearly as **Stakeholder Communication Templates**.

**Executive Summary**
Write this like a board presentation slide. Structure it as:

HEADLINE: One sentence leading with the total dollar opportunity.

SITUATION: One sentence describing the current state with a specific metric.

COMPLICATION: One sentence on why this is urgent — the cost of inaction.

RESOLUTION: One sentence on the recommended path forward.

KEY METRICS (show exactly 3 numbers that tell the story):
- [Metric 1]: [Value] — [what it means]
- [Metric 2]: [Value] — [what it means]
- [Metric 3]: [Value] — [what it means]

THE ASK: One specific request for resources, budget approval, or decision needed.

NEXT STEP: One specific action with a suggested timeframe (e.g. "Approve $X pilot budget by [timeframe] to begin Phase 1 within 30 days").

Label this section clearly as **Executive Summary** and put it at the END.
Do NOT include "Executive Summary**" as the first line — start directly with HEADLINE:

IMPORTANT RULES:
- Use specific numbers from the analysis everywhere — never say "significant" or "substantial" without a number
- Every initiative must have a named owner role
- Every KPI must have a baseline and target value
- The Executive Summary must lead with a dollar figure
- Write for a VP or C-suite audience — confident, direct, no hedging
- Avoid data science jargon entirely
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text


def generate_roi_charts(recommendations: str, api_key: str) -> list:
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
metric_label should describe what is being compared (e.g. "Annual Revenue", "Cost Savings").
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

        bars1 = ax.bar(x - width/2, current_vals, width, label='Current',
                       color='#64748b', edgecolor='white', alpha=0.8)
        bars2 = ax.bar(x + width/2, projected_vals, width,
                       label='Projected After Initiative',
                       color='#4ade80', edgecolor='white', alpha=0.9)

        ax.set_ylabel(initiatives[0].get("metric_label", "Value"), color='white', fontsize=12)
        ax.set_title('Projected Impact of Recommended Initiatives',
                     color='white', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right', color='white', fontsize=10)
        ax.tick_params(colors='white')
        ax.legend(facecolor='#1e293b', labelcolor='white', edgecolor='#334155')

        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:,.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', color='white', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:,.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', color='#4ade80', fontsize=9, fontweight='bold')

        for spine in ax.spines.values():
            spine.set_color('#334155')

        plt.tight_layout()
        plt.savefig('roi_chart_1.png', bbox_inches='tight',
                    facecolor='#1e293b', edgecolor='none')
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

        bars = ax.barh(labels, paybacks,
                       color=colors[:len(labels)], edgecolor='white', alpha=0.85)

        ax.set_xlabel('Payback Period (months)', color='white', fontsize=12)
        ax.set_title('Estimated Payback Period by Initiative',
                     color='white', fontsize=14, fontweight='bold')
        ax.tick_params(colors='white')

        for bar, val in zip(bars, paybacks):
            ax.annotate(f'{val} mo',
                        xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0), textcoords="offset points",
                        va='center', color='white', fontsize=10, fontweight='bold')

        for spine in ax.spines.values():
            spine.set_color('#334155')

        plt.tight_layout()
        plt.savefig('roi_chart_2.png', bbox_inches='tight',
                    facecolor='#1e293b', edgecolor='none')
        plt.close(fig)

        with open('roi_chart_2.png', 'rb') as f:
            charts.append(base64.b64encode(f.read()).decode('utf-8'))
        os.remove('roi_chart_2.png')
    except Exception as e:
        print(f"ROI chart 2 failed: {e}")

    plt.rcParams.update(plt.rcParamsDefault)
    return charts

def generate_monte_carlo(recommendations: str, api_key: str) -> dict:
    """
    Runs a Monte Carlo simulation on the projected savings/impact from the
    business recommendations. Returns percentile results, probability
    statements, and a histogram chart.
    """
    client = genai.Client(api_key=api_key)

    prompt = f"""
Extract the financial impact assumptions from this business recommendations text
so they can be used as inputs to a Monte Carlo simulation.

RECOMMENDATIONS TEXT:
{recommendations}

Return ONLY a JSON object with no extra text in this exact format:
{{
  "base_annual_savings": 722568,
  "low_estimate": 289027,
  "high_estimate": 1315625,
  "metric_label": "Annual Savings",
  "key_uncertain_variables": [
    {{"name": "Adoption rate of recommended changes", "low_pct": 60, "likely_pct": 85, "high_pct": 100}},
    {{"name": "Implementation timeline impact", "low_pct": 70, "likely_pct": 100, "high_pct": 110}},
    {{"name": "Magnitude of root cause reduction", "low_pct": 50, "likely_pct": 100, "high_pct": 130}}
  ]
}}

Use the actual conservative/base/aggressive numbers from the What-If Scenarios section if present.
key_uncertain_variables should reflect 2-4 real sources of uncertainty mentioned or implied in the text
(e.g. adoption rate, execution risk, timeline, magnitude of improvement).
low_pct/likely_pct/high_pct represent the variable as a percentage multiplier (100 = no change).
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    try:
        text = response.text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        inputs = json.loads(text)
    except Exception as e:
        print(f"Monte Carlo input extraction failed: {e}")
        return {}

    try:
        base = float(inputs.get("base_annual_savings", 0))
        low = float(inputs.get("low_estimate", base * 0.4))
        high = float(inputs.get("high_estimate", base * 1.8))
        metric_label = inputs.get("metric_label", "Annual Savings")
        variables = inputs.get("key_uncertain_variables", [])

        if base <= 0:
            return {}

        n_runs = 10000
        rng = np.random.default_rng(42)

        # Build a triangular distribution per uncertain variable, multiply together
        multiplier = np.ones(n_runs)
        for var in variables[:4]:
            low_pct = float(var.get("low_pct", 70)) / 100
            likely_pct = float(var.get("likely_pct", 100)) / 100
            high_pct = float(var.get("high_pct", 120)) / 100
            low_pct, high_pct = min(low_pct, high_pct), max(low_pct, high_pct)
            likely_pct = min(max(likely_pct, low_pct), high_pct)
            sampled = rng.triangular(low_pct, likely_pct, high_pct, n_runs)
            multiplier *= sampled

        # Normalize multiplier so its mean approximates 1.0 scaled to base
        multiplier = multiplier / np.mean(multiplier)
        simulated = base * multiplier

        # Clip to a sane range based on extracted low/high estimates with some tolerance
        simulated = np.clip(simulated, low * 0.5, high * 1.5)

        p5 = float(np.percentile(simulated, 5))
        p50 = float(np.percentile(simulated, 50))
        p95 = float(np.percentile(simulated, 95))
        mean_val = float(np.mean(simulated))
        std_val = float(np.std(simulated))

        prob_exceed_base = float(np.mean(simulated >= base)) * 100
        prob_exceed_low = float(np.mean(simulated >= low)) * 100
        prob_positive = float(np.mean(simulated > 0)) * 100

        # Generate histogram chart
        chart_b64 = ""
        try:
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1e293b')
            ax.set_facecolor('#0f172a')

            ax.hist(simulated, bins=50, color='#6366f1', alpha=0.8, edgecolor='none')
            ax.axvline(p5, color='#f87171', linestyle='--', linewidth=2, label=f'5th percentile: ${p5:,.0f}')
            ax.axvline(p50, color='#4ade80', linestyle='-', linewidth=2.5, label=f'Median: ${p50:,.0f}')
            ax.axvline(p95, color='#fbbf24', linestyle='--', linewidth=2, label=f'95th percentile: ${p95:,.0f}')

            ax.set_xlabel(metric_label, color='white', fontsize=12)
            ax.set_ylabel('Number of simulated outcomes', color='white', fontsize=12)
            ax.set_title(f'Monte Carlo Simulation — {n_runs:,} runs', color='white', fontsize=14, fontweight='bold')
            ax.tick_params(colors='white')
            ax.legend(facecolor='#1e293b', labelcolor='white', edgecolor='#334155', fontsize=10)

            for spine in ax.spines.values():
                spine.set_color('#334155')

            plt.tight_layout()
            plt.savefig('monte_carlo.png', bbox_inches='tight', facecolor='#1e293b', edgecolor='none')
            plt.close(fig)

            with open('monte_carlo.png', 'rb') as f:
                chart_b64 = base64.b64encode(f.read()).decode('utf-8')
            os.remove('monte_carlo.png')
            plt.rcParams.update(plt.rcParamsDefault)
        except Exception as e:
            print(f"Monte Carlo chart generation failed: {e}")

        result = {
            "metric_label": metric_label,
            "n_runs": n_runs,
            "p5": p5,
            "p50": p50,
            "p95": p95,
            "mean": mean_val,
            "std": std_val,
            "prob_exceed_base_pct": round(prob_exceed_base, 1),
            "prob_exceed_low_pct": round(prob_exceed_low, 1),
            "prob_positive_pct": round(prob_positive, 1),
            "base_estimate": base,
            "chart": chart_b64
        }
        print(f"Monte Carlo simulation completed: median=${p50:,.0f}, p5=${p5:,.0f}, p95=${p95:,.0f}")
        return result
    except Exception as e:
        print(f"Monte Carlo simulation failed: {e}")
        return {}

def generate_board_deck(summary: str, recommendations: str, api_key: str) -> dict:
    """
    Generates a 3-slide board-ready deck: the finding, the recommendation, the scenarios.
    Returns structured content plus one chart per slide.
    """
    client = genai.Client(api_key=api_key)

    prompt = f"""
You are preparing a 60-second board presentation based on this analysis.

ANALYSIS SUMMARY:
{summary}

BUSINESS RECOMMENDATIONS:
{recommendations}

Generate content for exactly 3 slides. Use EXACTLY this delimiter format, no JSON, no markdown fences:

===SLIDE1_HEADLINE===
[One sentence, the single biggest finding, plain English, under 16 words]
===SLIDE1_METRIC1_VAL===
[short value e.g. "5.7 hrs"]
===SLIDE1_METRIC1_LBL===
[short label e.g. "Avg dwell time vs 2.5 hr target"]
===SLIDE1_METRIC2_VAL===
[short value]
===SLIDE1_METRIC2_LBL===
[short label]
===SLIDE1_METRIC3_VAL===
[short value, ideally a dollar figure]
===SLIDE1_METRIC3_LBL===
[short label]
===SLIDE1_CHART_TITLE===
[chart title, e.g. "Top contributors to the problem"]
===SLIDE1_CHART_LABELS===
[comma separated category names, 3-5 categories, ordered highest to lowest]
===SLIDE1_CHART_VALUES===
[comma separated numeric values matching the labels, same order]

===SLIDE2_HEADLINE===
[One sentence framing the recommendation, under 16 words]
===SLIDE2_REC1===
[Initiative 1 name, under 8 words]
===SLIDE2_REC1_IMPACT===
[dollar impact, short, e.g. "+$361K/yr"]
===SLIDE2_REC2===
[Initiative 2 name, under 8 words]
===SLIDE2_REC2_IMPACT===
[dollar impact, short]
===SLIDE2_REC3===
[Initiative 3 name, under 8 words]
===SLIDE2_REC3_IMPACT===
[dollar impact, short]
===SLIDE2_ASK===
[One sentence: the budget ask and the ROI/payback, e.g. "Approve $600K to capture $722K in year-one savings — 120% ROI, 10 month payback"]

===SLIDE3_HEADLINE===
[One sentence framing the range of outcomes, under 16 words]
===SLIDE3_CONSERVATIVE_VAL===
[dollar value, short]
===SLIDE3_CONSERVATIVE_LBL===
[short label, e.g. "Quick wins only"]
===SLIDE3_BASE_VAL===
[dollar value, short]
===SLIDE3_BASE_LBL===
[short label, e.g. "Recommended path"]
===SLIDE3_AGGRESSIVE_VAL===
[dollar value, short]
===SLIDE3_AGGRESSIVE_LBL===
[short label, e.g. "Full transformation"]
===END===

Rules:
- Every number must come from the actual analysis — no placeholders
- Keep every text field SHORT — this is a slide, not a paragraph
- Slide 1 chart should show whatever breakdown best explains the root cause (defect categories, segments, regions, etc.)
- Use plain English, zero jargon, zero column names with underscores
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    text = response.text.strip()

    def extract(start_marker, end_marker, source):
        try:
            start_idx = source.index(start_marker) + len(start_marker)
            end_idx = source.index(end_marker, start_idx)
            return source[start_idx:end_idx].strip()
        except ValueError:
            return ""

    markers = [
        "SLIDE1_HEADLINE", "SLIDE1_METRIC1_VAL", "SLIDE1_METRIC1_LBL",
        "SLIDE1_METRIC2_VAL", "SLIDE1_METRIC2_LBL", "SLIDE1_METRIC3_VAL", "SLIDE1_METRIC3_LBL",
        "SLIDE1_CHART_TITLE", "SLIDE1_CHART_LABELS", "SLIDE1_CHART_VALUES",
        "SLIDE2_HEADLINE", "SLIDE2_REC1", "SLIDE2_REC1_IMPACT",
        "SLIDE2_REC2", "SLIDE2_REC2_IMPACT", "SLIDE2_REC3", "SLIDE2_REC3_IMPACT", "SLIDE2_ASK",
        "SLIDE3_HEADLINE", "SLIDE3_CONSERVATIVE_VAL", "SLIDE3_CONSERVATIVE_LBL",
        "SLIDE3_BASE_VAL", "SLIDE3_BASE_LBL", "SLIDE3_AGGRESSIVE_VAL", "SLIDE3_AGGRESSIVE_LBL"
    ]

    values = {}
    try:
        for i, marker in enumerate(markers):
            start = f"==={marker}==="
            end = f"==={markers[i+1]}===" if i + 1 < len(markers) else "===END==="
            values[marker] = extract(start, end, text)

        # Generate Slide 1 chart
        chart_b64 = ""
        try:
            labels = [l.strip() for l in values["SLIDE1_CHART_LABELS"].split(",") if l.strip()]
            vals = [float(v.strip().replace("%", "")) for v in values["SLIDE1_CHART_VALUES"].split(",") if v.strip()]

            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(9, 4), facecolor='#1e293b')
            ax.set_facecolor('#1e293b')

            colors = ['#f87171', '#fb923c', '#64748b', '#64748b', '#64748b'][:len(labels)]
            bars = ax.bar(labels, vals, color=colors, edgecolor='none', width=0.6)

            for bar, val in zip(bars, vals):
                ax.annotate(f'{val:g}%', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 6), textcoords="offset points",
                            ha='center', color='white', fontsize=13, fontweight='bold')

            ax.set_title(values.get("SLIDE1_CHART_TITLE", ""), color='white', fontsize=13, pad=16)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.tick_params(colors='white', labelsize=11)

            plt.tight_layout()
            plt.savefig('board_slide1.png', bbox_inches='tight', facecolor='#1e293b', edgecolor='none')
            plt.close(fig)

            with open('board_slide1.png', 'rb') as f:
                chart_b64 = base64.b64encode(f.read()).decode('utf-8')
            os.remove('board_slide1.png')
            plt.rcParams.update(plt.rcParamsDefault)
        except Exception as e:
            print(f"Board deck chart generation failed: {e}")

        result = {
            "slide1": {
                "headline": values["SLIDE1_HEADLINE"],
                "metrics": [
                    {"val": values["SLIDE1_METRIC1_VAL"], "lbl": values["SLIDE1_METRIC1_LBL"]},
                    {"val": values["SLIDE1_METRIC2_VAL"], "lbl": values["SLIDE1_METRIC2_LBL"]},
                    {"val": values["SLIDE1_METRIC3_VAL"], "lbl": values["SLIDE1_METRIC3_LBL"]},
                ],
                "chart": chart_b64
            },
            "slide2": {
                "headline": values["SLIDE2_HEADLINE"],
                "recommendations": [
                    {"text": values["SLIDE2_REC1"], "impact": values["SLIDE2_REC1_IMPACT"]},
                    {"text": values["SLIDE2_REC2"], "impact": values["SLIDE2_REC2_IMPACT"]},
                    {"text": values["SLIDE2_REC3"], "impact": values["SLIDE2_REC3_IMPACT"]},
                ],
                "ask": values["SLIDE2_ASK"]
            },
            "slide3": {
                "headline": values["SLIDE3_HEADLINE"],
                "scenarios": [
                    {"label": "Conservative", "val": values["SLIDE3_CONSERVATIVE_VAL"], "sub": values["SLIDE3_CONSERVATIVE_LBL"]},
                    {"label": "Base case", "val": values["SLIDE3_BASE_VAL"], "sub": values["SLIDE3_BASE_LBL"]},
                    {"label": "Aggressive", "val": values["SLIDE3_AGGRESSIVE_VAL"], "sub": values["SLIDE3_AGGRESSIVE_LBL"]},
                ]
            }
        }
        print(f"Board deck generated successfully: headline='{result['slide1']['headline'][:50]}'")
        return result
    except Exception as e:
        print(f"Board deck generation failed: {e}")
        print(f"Raw response (first 500 chars): {text[:500]}")
        return {}

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
  "analysis_type": "regression|classification|clustering|anomaly|timeseries|sixsigma",
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
    "dry_van": null, "reefer": null, "flatbed": null,
    "step_deck": null, "lowboy": null, "tanker": null,
    "pup_28": null, "pup_doubles": null, "conestoga": null,
    "double_drop": null, "rgn": null, "liftgate_van": null,
    "reefer_liftgate": null, "curtain_side": null,
    "intermodal": null, "pneumatic": null
  }},
  "last_updated": "2025-06-15",
  "dataset_rows": {len(df)},
  "dataset_columns": {len(df.columns)}
}}

Fill in values from the analysis where available.
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
            "last_updated": "2025-06-15",
            "dataset_rows": len(df),
            "dataset_columns": len(df.columns)
        }


def generate_sql(summary: str, df, filename: str, api_key: str) -> dict:
    """
    Generates production-ready SQL queries based on the analysis.
    Returns a dict with multiple SQL sections.
    """
    client = genai.Client(api_key=api_key)

    col_types = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        if 'int' in dtype:
            sql_type = 'INTEGER'
        elif 'float' in dtype:
            sql_type = 'DECIMAL(10,2)'
        elif 'datetime' in dtype:
            sql_type = 'DATETIME'
        elif 'bool' in dtype:
            sql_type = 'BOOLEAN'
        else:
            sql_type = 'VARCHAR(255)'
        col_types.append(f"    {col} {sql_type}")

    table_name = filename.replace('.csv', '').replace('-', '_').replace(' ', '_').lower()

    prompt = f"""
You are a senior data engineer. Based on this analysis summary, generate production-ready SQL queries.

TABLE NAME: {table_name}
COLUMNS AND TYPES:
{chr(10).join(col_types)}

ANALYSIS SUMMARY:
{summary}

Generate SQL in this exact JSON format with no extra text:
{{
  "table_name": "{table_name}",
  "create_table": "CREATE TABLE {table_name} (\\n{chr(10).join(col_types)}\\n);",
  "main_query": "Primary SELECT query that pulls the exact data used in the analysis",
  "metric_queries": [
    {{
      "title": "Query title",
      "description": "What this query does in plain English",
      "sql": "SELECT statement here"
    }}
  ],
  "monitoring_query": {{
    "title": "Daily Monitoring Query",
    "description": "Run this daily or weekly to track key metrics over time",
    "sql": "SELECT statement here"
  }}
}}

Rules:
- Use standard ANSI SQL compatible with SQL Server, Snowflake, BigQuery, PostgreSQL
- Add SQL comments (--) explaining each section
- Generate 3-5 metric queries, one per major finding from the analysis
- The monitoring query should aggregate the most important metrics by date
- Use the actual column names from the dataset exactly as listed above
- Include WHERE clauses that mirror the filters used in the analysis
- Include ORDER BY clauses appropriate for each query
- For financial impact queries include cost calculations where relevant
- Keep each SQL query clean, readable, and well-commented
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    try:
        text = response.text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        print(f"SQL generated successfully for table: {result.get('table_name', 'unknown')}")
        return result
    except Exception as e:
        print(f"SQL generation failed: {e}")
        print(f"Raw response: {response.text[:500]}")
        return {
            "table_name": table_name,
            "create_table": f"-- Could not generate CREATE TABLE for {table_name}",
            "main_query": "-- Could not generate main query",
            "metric_queries": [],
            "monitoring_query": {
                "title": "Daily Monitoring Query",
                "description": "Could not generate monitoring query",
                "sql": "-- Could not generate monitoring query"
            }
        }

def generate_vba(summary: str, df, filename: str, api_key: str) -> dict:
    """
    Generates Excel VBA macros based on the analysis.
    Uses delimiter-based parsing instead of JSON to avoid quote-escaping issues with VBA code.
    """
    client = genai.Client(api_key=api_key)

    col_list = list(df.columns)
    sheet_name = filename.replace('.csv', '').replace('.xlsx', '').replace('.xls', '').replace('.json', '').replace('.pdf', '').replace('-', '_').replace(' ', '_')

    prompt = f"""
You are a senior Excel/VBA developer. Based on this analysis summary, generate practical VBA macros
that a business user could run in Excel to replicate and monitor the key findings.

DATASET COLUMNS: {col_list}
SHEET NAME REFERENCE: {sheet_name}

ANALYSIS SUMMARY:
{summary}

Generate 5 VBA macros. Use EXACTLY this format with no extra text, no markdown code fences, 
no JSON — just plain text with these exact delimiters:

===WORKBOOK_SETUP===
[Complete VBA Sub for workbook/sheet setup here]

===KPI_TITLE===
KPI Dashboard Macro
===KPI_DESCRIPTION===
[One sentence plain English description]
===KPI_VBA===
[Complete VBA Sub here]

===ANOMALY_TITLE===
Anomaly Flagging Macro
===ANOMALY_DESCRIPTION===
[One sentence plain English description]
===ANOMALY_VBA===
[Complete VBA Sub here]

===CHART_TITLE===
Chart Generation Macro
===CHART_DESCRIPTION===
[One sentence plain English description]
===CHART_VBA===
[Complete VBA Sub here]

===REFRESH_TITLE===
Data Refresh Macro
===REFRESH_DESCRIPTION===
[One sentence plain English description]
===REFRESH_VBA===
[Complete VBA Sub here]
===END===

Rules:
- Write complete, working VBA — Sub/End Sub structure, proper variable declarations
- Add VBA comments (') explaining each section
- Reference actual column names from the dataset where relevant
- Use realistic cell references and ranges based on the column count
- The KPI dashboard macro should calculate the same key metrics found in the analysis
- The anomaly macro should use the same thresholds/logic described in the analysis
- The chart macro should chart the single most important relationship from the analysis
- Keep macros practical — a business analyst with basic VBA knowledge should be able to use these
- Do not use any external libraries or references beyond standard Excel VBA
- Do NOT wrap your response in markdown code fences or JSON — use only the delimiters shown above
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    text = response.text.strip()

    def extract_between(start_marker, end_marker, source):
        try:
            start_idx = source.index(start_marker) + len(start_marker)
            end_idx = source.index(end_marker, start_idx)
            return source[start_idx:end_idx].strip()
        except ValueError:
            return ""

    try:
        workbook_setup = extract_between("===WORKBOOK_SETUP===", "===KPI_TITLE===", text)

        kpi_title = extract_between("===KPI_TITLE===", "===KPI_DESCRIPTION===", text)
        kpi_desc = extract_between("===KPI_DESCRIPTION===", "===KPI_VBA===", text)
        kpi_vba = extract_between("===KPI_VBA===", "===ANOMALY_TITLE===", text)

        anomaly_title = extract_between("===ANOMALY_TITLE===", "===ANOMALY_DESCRIPTION===", text)
        anomaly_desc = extract_between("===ANOMALY_DESCRIPTION===", "===ANOMALY_VBA===", text)
        anomaly_vba = extract_between("===ANOMALY_VBA===", "===CHART_TITLE===", text)

        chart_title = extract_between("===CHART_TITLE===", "===CHART_DESCRIPTION===", text)
        chart_desc = extract_between("===CHART_DESCRIPTION===", "===CHART_VBA===", text)
        chart_vba = extract_between("===CHART_VBA===", "===REFRESH_TITLE===", text)

        refresh_title = extract_between("===REFRESH_TITLE===", "===REFRESH_DESCRIPTION===", text)
        refresh_desc = extract_between("===REFRESH_DESCRIPTION===", "===REFRESH_VBA===", text)
        refresh_vba = extract_between("===REFRESH_VBA===", "===END===", text)

        result = {
            "workbook_setup": workbook_setup or "' Could not extract workbook setup macro",
            "kpi_dashboard_macro": {
                "title": kpi_title or "KPI Dashboard Macro",
                "description": kpi_desc,
                "vba": kpi_vba or "' Could not extract KPI dashboard macro"
            },
            "anomaly_flagging_macro": {
                "title": anomaly_title or "Anomaly Flagging Macro",
                "description": anomaly_desc,
                "vba": anomaly_vba or "' Could not extract anomaly flagging macro"
            },
            "chart_macro": {
                "title": chart_title or "Chart Generation Macro",
                "description": chart_desc,
                "vba": chart_vba or "' Could not extract chart macro"
            },
            "refresh_macro": {
                "title": refresh_title or "Data Refresh Macro",
                "description": refresh_desc,
                "vba": refresh_vba or "' Could not extract refresh macro"
            }
        }
        print("VBA generated successfully using delimiter parsing")
        return result
    except Exception as e:
        print(f"VBA generation failed: {e}")
        print(f"Raw response (first 1000 chars): {text[:1000]}")
        return {
            "workbook_setup": "' Could not generate workbook setup macro",
            "kpi_dashboard_macro": {
                "title": "KPI Dashboard Macro",
                "description": "Could not generate",
                "vba": "' Could not generate KPI dashboard macro"
            },
            "anomaly_flagging_macro": {
                "title": "Anomaly Flagging Macro",
                "description": "Could not generate",
                "vba": "' Could not generate anomaly flagging macro"
            },
            "chart_macro": {
                "title": "Chart Generation Macro",
                "description": "Could not generate",
                "vba": "' Could not generate chart macro"
            },
            "refresh_macro": {
                "title": "Data Refresh Macro",
                "description": "Could not generate",
                "vba": "' Could not generate refresh macro"
            }
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


def run_agent(goal: str, df, api_key: str, filename: str = "dataset.csv", max_turns: int = 8):
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
        if reply is None:
            print("Gemini returned None response, retrying turn...")
            messages.append(user_msg("Your last response was empty. Please continue your analysis with a code block or FINAL ANSWER."))
            continue

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
    
    monte_carlo = {}
    if final_summary and recommendations:
        monte_carlo = generate_monte_carlo(recommendations, api_key)

    confidence_scores = {}
    if final_summary:
        confidence_scores = get_confidence_scores(final_summary, api_key)

    devils_advocate = ""
    if final_summary:
        devils_advocate = get_devils_advocate(final_summary, recommendations, api_key)

    model_export = {}
    if final_summary:
        model_export = extract_model_export(final_summary, df, api_key)

    sql_queries = {}
    if final_summary:
        sql_queries = generate_sql(final_summary, df, filename, api_key)

    vba_macros = {}
    if final_summary:
        vba_macros = generate_vba(final_summary, df, filename, api_key)

    board_deck = {}
    if final_summary:
        board_deck = generate_board_deck(final_summary, recommendations, api_key)

    return {
        "summary": final_summary,
        "charts": all_charts,
        "roi_charts": roi_charts,
        "plotly_charts": [],
        "turns": len(messages),
        "recommendations": recommendations,
        "quality_report": quality_report,
        "devils_advocate": devils_advocate,
        "confidence_scores": confidence_scores,
        "model_export": model_export,
        "sql_queries": sql_queries,
        "vba_macros": vba_macros,
        "board_deck": board_deck,
        "monte_carlo": monte_carlo
    }
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
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import shap
import shap
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, shapiro
from scipy.stats import f_oneway, kruskal, spearmanr, pearsonr
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import learning_curve
from sklearn.feature_selection import RFE, SelectKBest, f_classif, f_regression

SYSTEM_INSTRUCTION = """
You are an expert data scientist agent.
You are given a dataset and a goal. You reason step by step like a senior data scientist would.

## Your workflow:
1. INSPECT the data — column types, nulls, distributions, target variable
1.5. ENGINEER FEATURES before modeling:
   - Create ratio features (e.g. revenue_per_customer = revenue / customers)
   - Create interaction features for highly correlated numeric columns
   - Bin continuous variables into meaningful categories (e.g. age groups)
   - Create lag features if date column exists (previous period values)
   - For text-like categorical columns with many values: group rare categories into "Other"
   - Print which new features were created and why
   - Always explain how engineered features improved the model
   - After feature engineering, run RFE (Recursive Feature Elimination):
     * Use RFE with the primary model and n_features_to_select=10 (or all if fewer than 10)
     * Print which features were selected and which were eliminated
     * Print the ranking of eliminated features (1 = most important)
     * Use only the selected features for final model training
     * Compare model performance with all features vs RFE-selected features
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
   - Dataset has two groups/variants (A/B, control/treatment, before/after) → A/B Test Analysis:
     * Identify the control and treatment groups
     * For conversion metrics (binary): use chi-square test
     * For continuous metrics (revenue, time): use t-test or Mann-Whitney U test
     * Calculate statistical significance (p-value) and confidence intervals
     * Calculate relative uplift (% improvement of treatment over control)
     * Determine minimum sample size needed for reliable results
     * State clearly: is the result statistically significant? (p < 0.05)
     * Generate a bar chart comparing groups with error bars
     * Give a plain English recommendation: ship it, don't ship it, or need more data
   - Explicit statistical testing request → Hypothesis Testing:
     * Normality check: use Shapiro-Wilk test on each group
     * Two groups comparison: use t-test (normal data) or Mann-Whitney U (non-normal)
     * Multiple groups comparison: use ANOVA (normal) or Kruskal-Wallis (non-normal)
     * Categorical relationships: use chi-square test
     * Correlation analysis: use Pearson (normal) or Spearman (non-normal)
     * Always state null hypothesis, alternative hypothesis, p-value, and conclusion
     * Use p < 0.05 as significance threshold unless specified otherwise
     * Generate visualizations: box plots, distribution plots, correlation heatmap
3. SELECT the best 2-3 models for the problem and explain why
   - For large datasets (>5000 rows): prefer LightGBM or XGBoost (faster)
   - For small datasets (<1000 rows): consider SVM or Logistic Regression
   - Always compare at least 2 models
4. CONSIDER hyperparameters — don't just use defaults, reason about:
   - For Logistic Regression: C (regularization strength), max_iter, solver
   - For Random Forest: n_estimators, max_depth, min_samples_split
   - For Linear Regression: whether to normalize, handle outliers
   - For K-Means: n_clusters (use elbow method)
5. TRAIN the model using the appropriate validation strategy:
   - If dataset has LESS than 1000 rows: use 5-fold cross-validation
     * Use cross_val_score for each metric (accuracy, precision, recall, F1)
     * Report mean ± standard deviation for each metric
     * A low std (< 0.05) means stable model, high std (> 0.10) means unstable/overfit
     * Also do a final fit on full training data for SHAP analysis
   - If dataset has MORE than 1000 rows: use 80/20 train/test split as normal
   - Always print which validation strategy was used and why
   - CHECK FOR CLASS IMBALANCE before training:
     * Calculate the ratio of majority to minority class
     * If imbalance ratio > 2:1, apply SMOTE to the training data only
     * Print class distribution before and after SMOTE
     * Never apply SMOTE to test data
     * Report whether SMOTE was applied and how it affected class distribution
6. EVALUATE using the right metrics:
   - Classification → accuracy, precision, recall, F1, confusion matrix
   - Regression → RMSE, MAE, R² score
7. GENERATE SHAP values to explain model predictions:
   - Use shap.TreeExplainer for Random Forest models
   - Use shap.LinearExplainer for Logistic/Linear Regression
   - Print the top 5 most impactful features with their average SHAP values
   - Generate a SHAP feature importance bar chart using matplotlib
8. VISUALIZE results using matplotlib:
   - Always save charts with plt.savefig('chart.png', bbox_inches='tight'); plt.close()
   - Never use plt.show()
   - Generate confusion matrix, feature importance, and SHAP plots
   - After training, generate a learning curve for the best model:
     * Use learning_curve() with cv=5 and train_sizes=np.linspace(0.1, 1.0, 10)
     * Plot training score and cross-validation score vs training set size
     * Use plt.fill_between() to show the variance band around each curve
     * A large gap between training and CV score = overfitting
     * Both curves flat and low = underfitting (needs more features)
     * Both curves converging high = good fit
     * Save as chart.png and explain what the curve reveals about the model
9. SUMMARIZE findings in plain English including SHAP explanations

## Rules:
- Always wrap code in ```python ... ``` blocks
- Generate ALL charts in SEPARATE code blocks BEFORE writing FINAL ANSWER
- Always save every chart with plt.savefig('chart.png', bbox_inches='tight'); plt.close()
- Always set figure background: plt.figure(figsize=(10, 6), facecolor='#1e293b')
- Always set axes background: ax.set_facecolor('#0f172a') or plt.gca().set_facecolor('#0f172a')
- Never use plt.show()
- Never mention chart filenames in text — always generate them in actual code blocks
- Encode categorical variables before modeling
- Always print evaluation metrics clearly
- For anomaly detection: always print how many anomalies were found and what % of the dataset they represent
- For anomaly detection: always explain the top 3 anomalies in plain English
- For trend analysis: always generate at least 2 charts — a time series line chart and a growth rate bar chart
- For trend analysis: always state the trend direction clearly and what it means for the business
- For trend analysis: always save charts with plt.savefig('chart.png', bbox_inches='tight'); plt.close()
- Always attempt basic feature engineering before modeling — create at least 1-2 new features using pd.cut, pd.qcut, ratios, or interactions
- Print the correlation matrix of top features before modeling to identify engineering opportunities
- After modeling, compare performance with and without engineered features if time permits
- The FINAL ANSWER section should contain only plain text summary, no code blocks
- When done, start your final message with: FINAL ANSWER:
"""


def get_ml_tools(df):
    """Returns all tools available to the agent during code execution."""
    return {
        "pd": pd, "plt": plt, "sns": sns, "os": os, "np": np,
        "df": df,
        "shap": shap,
        "LogisticRegression": LogisticRegression,
        "LinearRegression": LinearRegression,
        "RandomForestClassifier": RandomForestClassifier,
        "RandomForestRegressor": RandomForestRegressor,
        "KMeans": KMeans,
        "train_test_split": train_test_split,
        "GridSearchCV": GridSearchCV,
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
        "XGBClassifier": XGBClassifier,
        "XGBRegressor": XGBRegressor,
        "LGBMClassifier": LGBMClassifier,
        "LGBMRegressor": LGBMRegressor,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "SVC": SVC,
        "SVR": SVR,
        "Ridge": Ridge,
        "Lasso": Lasso,
        "IsolationForest": IsolationForest,
        "DBSCAN": DBSCAN,
        "cross_val_score": cross_val_score,
        "StratifiedKFold": StratifiedKFold,
        "KFold": KFold,
        "stats": stats,
        "chi2_contingency": chi2_contingency,
        "ttest_ind": ttest_ind,
        "mannwhitneyu": mannwhitneyu,
        "shapiro": shapiro,
        "f_oneway": f_oneway,
        "kruskal": kruskal,
        "spearmanr": spearmanr,
        "pearsonr": pearsonr,
        "pd": pd,  # already there, but ensure cut and qcut are available
        "cut": pd.cut,
        "qcut": pd.qcut,
        "SMOTE": SMOTE,
        "learning_curve": learning_curve,
        "RFE": RFE,
        "SelectKBest": SelectKBest,
        "f_classif": f_classif,
        "f_regression": f_regression,
    }


def run_python(code: str, df) -> tuple[str, list[str]]:
    """
    Executes Python code and returns (text_output, list_of_base64_charts).
    """
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    charts = []

    try:
        # Set dark background for all charts
        plt.style.use('dark_background')
        plt.rcParams['figure.facecolor'] = '#1e293b'
        plt.rcParams['axes.facecolor'] = '#0f172a'
        plt.rcParams['savefig.facecolor'] = '#1e293b'
        plt.rcParams['savefig.edgecolor'] = 'none'

        exec(code, get_ml_tools(df))

        # Force-save any still-open matplotlib figures
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
- If the summary mentions predictions are unreliable or not better than guessing, overall confidence MUST be LOW
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
   - Estimated implementation cost (e.g. $10,000 - $50,000)
   - Estimated annual revenue impact or cost savings
   - ROI percentage (revenue impact / cost * 100)
   - Estimated payback period (e.g. 2 months, 6 months)
   Label this section clearly as **Initiative ROI Scorecard**
   Format as a markdown table with columns: Initiative | Est. Cost | Revenue Impact | ROI | Payback Period

4. **KPIs to Track** — How to measure success for each initiative

5. **Quick Wins** — 1-2 things that could be implemented immediately with low effort and high impact

6. **6 Month Roadmap** — A suggested timeline for implementing the initiatives in order

7. **What-If Scenarios** — 2-3 financial impact estimates.
   Label this section clearly as **What-If Scenarios**.

8. **Executive Summary** — A concise summary for C-suite leadership containing:
   - One sentence describing the single biggest opportunity
   - Exactly 3 bullet points of key findings in plain English
   - The total financial opportunity in one bold number (e.g. **$1.2M annual opportunity**)
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
    if final_summary:
        recommendations = get_business_recommendations(final_summary, api_key)

    confidence_scores = {}
    if final_summary:
        confidence_scores = get_confidence_scores(final_summary, api_key)

    return {
        "summary": final_summary,
        "charts": all_charts,
        "plotly_charts": [],
        "turns": len(messages),
        "recommendations": recommendations,
        "quality_report": quality_report,
        "confidence_scores": confidence_scores
    }
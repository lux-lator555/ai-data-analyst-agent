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

SYSTEM_INSTRUCTION = """
You are an expert data scientist agent.
You are given a dataset and a goal. You reason step by step like a senior data scientist would.

## Your workflow:
1. INSPECT the data — column types, nulls, distributions, target variable
2. IDENTIFY the problem type based on the target variable:
   - Binary column (0/1, yes/no) → Try Logistic Regression, Random Forest, XGBoost, LightGBM
   - Continuous numeric column → Try Linear Regression, Ridge, Lasso, XGBoost Regressor
   - Categorical column with 3+ classes → Try Random Forest, XGBoost, LightGBM
   - No clear target → K-Means Clustering or DBSCAN
   - Anomaly detection task → Isolation Forest
3. SELECT the best 2-3 models for the problem and explain why
   - For large datasets (>5000 rows): prefer LightGBM or XGBoost (faster)
   - For small datasets (<1000 rows): consider SVM or Logistic Regression
   - Always compare at least 2 models
4. CONSIDER hyperparameters — don't just use defaults, reason about:
   - For Logistic Regression: C (regularization strength), max_iter, solver
   - For Random Forest: n_estimators, max_depth, min_samples_split
   - For Linear Regression: whether to normalize, handle outliers
   - For K-Means: n_clusters (use elbow method)
5. TRAIN the model using an 80/20 train/test split
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
9. SUMMARIZE findings in plain English including SHAP explanations

## Rules:
- Always wrap code in ```python ... ``` blocks
- Generate ALL charts in SEPARATE code blocks BEFORE writing FINAL ANSWER
- Always save every chart with plt.savefig('chart.png', bbox_inches='tight'); plt.close()
- Never use plt.show()
- Never mention chart filenames in text — always generate them in actual code blocks
- Encode categorical variables before modeling
- Always print evaluation metrics clearly
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
    }


def run_python(code: str, df) -> tuple[str, list[str]]:
    """
    Executes Python code and returns (text_output, list_of_base64_charts).
    """
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    charts = []

    try:
        exec(code, get_ml_tools(df))

        for i, fig in enumerate(map(plt.figure, plt.get_fignums())):
            fig.savefig(f'chart_{i}.png', bbox_inches='tight')
            plt.close(fig)

        output = buffer.getvalue()

    except Exception as e:
        output = f"ERROR: {str(e)}"

    finally:
        sys.stdout = old_stdout

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
# agent.py — Core agent logic (converted from Colab notebook)

import io
import os
import sys
import time
import base64
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

SYSTEM_INSTRUCTION = """
You are an expert data scientist agent.
You are given a dataset and a goal. You reason step by step like a senior data scientist would.

## Your workflow:
1. INSPECT the data — column types, nulls, distributions, target variable
2. IDENTIFY the problem type based on the target variable:
   - Binary column (0/1, yes/no) → Logistic Regression + Random Forest (classification)
   - Continuous numeric column → Linear Regression (regression)
   - Categorical column with 3+ classes → Random Forest (multiclass classification)
   - No clear target → K-Means Clustering (unsupervised)
3. SELECT the best model and explain why
4. CONSIDER hyperparameters — don't just use defaults, reason about:
   - For Logistic Regression: C (regularization strength), max_iter, solver
   - For Random Forest: n_estimators, max_depth, min_samples_split
   - For Linear Regression: whether to normalize, handle outliers
   - For K-Means: n_clusters (use elbow method)
5. TRAIN the model using an 80/20 train/test split
6. EVALUATE using the right metrics:
   - Classification → accuracy, precision, recall, F1, confusion matrix
   - Regression → RMSE, MAE, R² score
7. VISUALIZE results — feature importance, confusion matrix, residual plots
8. SUMMARIZE findings in plain English

## Rules:
- Always wrap code in ```python ... ``` blocks
- Always save charts with plt.savefig('chart.png', bbox_inches='tight'); plt.close()
- Never use plt.show()
- Encode categorical variables before modeling
- Always print evaluation metrics clearly
- When done, start your final message with: FINAL ANSWER:
"""


def get_ml_tools(df):
    """Returns all tools available to the agent during code execution."""
    return {
        "pd": pd, "plt": plt, "sns": sns, "os": os, "np": np,
        "df": df,
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
    }


def run_python(code: str, df) -> tuple[str, list[str]]:
    """
    Executes Python code and returns (text_output, list_of_base64_charts).
    Charts are returned as base64 strings so the API can send them to the frontend.
    """
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    charts = []

    try:
        exec(code, get_ml_tools(df))

        # Force-save any still-open figures
        for i, fig in enumerate(map(plt.figure, plt.get_fignums())):
            fig.savefig(f'chart_{i}.png', bbox_inches='tight')
            plt.close(fig)

        output = buffer.getvalue()

    except Exception as e:
        output = f"ERROR: {str(e)}"

    finally:
        sys.stdout = old_stdout

    # Convert saved charts to base64 so they can be sent over the API
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


def user_msg(text):
    return types.Content(role="user", parts=[types.Part(text=text)])


def model_msg(text):
    return types.Content(role="model", parts=[types.Part(text=text)])


def run_agent(goal: str, df, api_key: str, max_turns: int = 8):
    """
    Main agent loop. Returns a dict with the final summary and any charts.
    """
    client = genai.Client(api_key=api_key)

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

        # Retry up to 3 times if Gemini is temporarily unavailable
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
            if "```python" in reply:
                code_block = reply.split("```python")[1].split("```")[0].strip()
                _, charts = run_python(code_block, df)
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

    return {
        "summary": final_summary,
        "charts": all_charts,
        "turns": len(messages)
    }
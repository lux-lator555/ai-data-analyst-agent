# ai-data-analyst-agent
AI Agent to analyze data and process models depending on the data. 
# 🤖 AI Data Analyst Agent

An autonomous data analysis agent powered by Google Gemini 2.5 Flash. Upload any CSV dataset and the agent will inspect the data, select the appropriate machine learning model, tune hyperparameters, train and evaluate the model, generate visualizations, and summarize findings in plain English — all autonomously.

---

## 🎯 What It Does

- Accepts any CSV dataset via file upload
- Inspects column types, distributions, and identifies the problem type
- Selects the appropriate ML model based on the data:
  - **Binary target** → Logistic Regression + Random Forest (classification)
  - **Continuous target** → Linear Regression (regression)
  - **Multi-class target** → Random Forest (multiclass classification)
  - **No clear target** → K-Means Clustering (unsupervised)
- Reasons about and selects hyperparameters (not just defaults)
- Trains models on an 80/20 train/test split
- Evaluates with the right metrics (accuracy, F1, RMSE, R², etc.)
- Generates charts: confusion matrices, ROC curves, feature importance, trend plots
- Produces a plain-English summary of all findings

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| AI Model | Google Gemini 2.5 Flash |
| Agent Framework | Custom Python agent loop |
| ML Library | scikit-learn |
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Environment | Google Colab |

---

## 🚀 Getting Started

### 1. Open in Google Colab
Click the notebook file above (`AI_Data_Analyst_Agent.ipynb`) and open it in Colab.

### 2. Get a free Gemini API key
- Go to [aistudio.google.com](https://aistudio.google.com)
- Click **Get API key** → **Create API key**
- No credit card required

### 3. Add your API key
In Cell 2 of the notebook, replace the placeholder:
your-api-key-here

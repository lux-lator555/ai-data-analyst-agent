# 🤖 AI Data Analyst Agent

An autonomous AI-powered data analysis platform that transforms any CSV dataset into actionable business intelligence. Upload your data, and the agent automatically detects the problem type, selects and trains the appropriate ML models, explains predictions with SHAP, and delivers ranked business recommendations with financial impact scenarios.

---

## 🎯 What It Does

### Intelligent Analysis
- **Auto-detects** dataset type and suggests the optimal analysis goal automatically
- **Data Quality Report** — flags missing values, outliers, and duplicates before modeling
- **Selects the right ML model** based on the data:
  - Binary target → Logistic Regression + Random Forest (classification)
  - Continuous target → Linear Regression (regression)
  - Multi-class target → Random Forest (multiclass classification)
  - No clear target → K-Means Clustering (unsupervised)
- **Reasons about hyperparameters** — doesn't just use defaults
- **SHAP Explainability** — explains *why* the model made each prediction

### Model Evaluation
- Trains on 80/20 train/test split
- Evaluates with the correct metrics per problem type (accuracy, F1, ROC-AUC, RMSE, R²)
- **Model Comparison Table** — side by side metrics with best model highlighted
- **Confidence Assessment** — 🟢🟡🔴 honest scoring of each finding with caveats

### Business Intelligence
- **Ranked Initiatives** — specific actions with effort, priority, and expected impact
- **KPIs to Track** — measurable success metrics for each initiative
- **Quick Wins** — immediate low-effort high-impact actions
- **6 Month Roadmap** — phased implementation timeline
- **What-If Scenarios** — financial impact estimates (e.g. "retaining 15% more at-risk customers = $345,000 additional revenue")

### User Experience
- 🤖 Walking robot progress indicator with rotating witty messages
- **Suggested prompt buttons** for common use cases
- **Analysis history** — click back to previous results within a session
- **Copy to clipboard** on business recommendations
- **PDF export** of the full report
- **Follow-up chat** — conversational Q&A with the agent about the results
- **Email authentication** via Cloudflare Access (one-time PIN)

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React PWA, Cloudflare Pages |
| Backend | FastAPI, Python, Render |
| AI Reasoning | Google Gemini 2.5 Flash |
| ML Models | scikit-learn |
| Explainability | SHAP |
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Auth | Cloudflare Access |
| Version Control | GitHub |

---

## 🚀 Getting Started

### 1. Open the live app
👉 **[ai-agent-frontend-2ij.pages.dev](https://ai-agent-frontend-2ij.pages.dev)**

> Access requires email verification via one-time PIN. Contact the owner to request access.

### 2. Get a free Gemini API key
- Go to [aistudio.google.com](https://aistudio.google.com)
- Click **Get API key** → **Create API key**
- No credit card required for the free tier

### 3. Run an analysis
1. Enter your Gemini API key
2. Upload any CSV file — the agent will auto-detect what to analyze
3. Review or edit the suggested goal
4. Click **▶ Run Agent**
5. Explore results, ask follow-up questions, download PDF

---

## 📊 Supported Analysis Types

| Use Case | Problem Type | Models |
|----------|-------------|--------|
| Customer churn prediction | Binary classification | Logistic Regression, Random Forest, XGBoost, LightGBM |
| Shipment delay prediction | Binary classification | Logistic Regression, Random Forest, XGBoost, LightGBM |
| Next product recommendation | Multiclass classification | Random Forest, XGBoost, LightGBM |
| Employee salary prediction | Regression | Linear Regression, Ridge, Lasso |
| Customer segmentation | Clustering | K-Means, DBSCAN |
| Fraud / anomaly detection | Anomaly detection | Isolation Forest, DBSCAN |
| Sales / revenue trends | Time series | Trend analysis, forecasting |
| Any binary yes/no outcome | Classification | Logistic Regression, Random Forest, XGBoost |

---

## 🧠 How the Agent Works
This differs from traditional ML pipelines because the agent **decides** what to do at each step based on what it finds — self-correcting if code fails and adapting its approach to the specific dataset.

---

## 📈 Example Output (Telco Churn Dataset — 7,043 rows)

**Auto-detected:** Binary classification → Churn prediction

**Models trained:** Logistic Regression (80.3% accuracy, ROC-AUC 0.84) + Random Forest (80.3% accuracy, ROC-AUC 0.84)

**Top SHAP features:** Contract type, tenure, monthly charges, internet service type, payment method

**Confidence:** 🟢 HIGH across all findings

**Financial scenarios generated:**
- Retaining 15% of at-risk customers → **$345,000 additional annual revenue**
- Converting 20% of month-to-month contracts → **$460,000 annual benefit**

---

## 🗺️ Roadmap

## 🗺️ Roadmap

- [x] Autonomous agent loop
- [x] CSV upload with auto-detection
- [x] ML model selection and hyperparameter reasoning
- [x] Cross-validation for small datasets (< 1000 rows)
- [x] SHAP explainability
- [x] Data quality report
- [x] Confidence scoring
- [x] Chart generation
- [x] Model comparison table
- [x] Business recommendations with ranked initiatives
- [x] Initiative ROI Scorecard
- [x] What-if financial scenarios
- [x] 6 month roadmap generation
- [x] Executive Summary with copy for presentation
- [x] Follow-up chat interface
- [x] PDF export
- [x] Analysis history
- [x] Email authentication
- [x] Anomaly detection (Isolation Forest + DBSCAN)
- [x] Trend analysis with forecasting and visualizations
- [x] XGBoost, LightGBM, SVM, Ridge, Lasso support
- [ ] Interactive Plotly charts
- [ ] Google Sheets connector
- [ ] Persistent history across sessions
- [ ] User accounts

---

## 🏗️ Architecture
---

## 📄 License
MIT

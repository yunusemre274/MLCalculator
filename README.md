# Machine Learning Calculator Web Application

**Full-Stack ML Platform** with React + TypeScript Frontend and FastAPI Python Backend

## 🎯 Project Overview

A comprehensive machine learning web application that provides:
- **Automated EDA** with interactive visualizations
- **79+ ML Models** (47 regression, 32 classification)
- **Hyperparameter Tuning** with GridSearchCV/RandomizedSearchCV
- **Smart Data Cleaning** with auto column removal
- **Interactive Visualizations** using Plotly
- **LazyPredict-style Results** with performance tables

---

## 🚀 Quick Start

### Option 1: Using BAT Files (Recommended for Windows)

**Start both servers:**
```bash
start.bat
```

**Stop both servers:**
```bash
stop.bat
```

### Option 2: Manual Start

**Terminal 1 - Backend:**
```bash
cd backend
uvicorn app.main:app --reload
```
Backend runs at: `http://localhost:8000`

**Terminal 2 - Frontend:**
```bash
npm run dev
```
Frontend runs at: `http://localhost:8080`

---

## 📋 Application Flow (STRICT OUTPUT PLACEMENT)

### **EDA Tab** - Exploratory Data Analysis ✅
Contains ALL visualizations and tables:
- 📊 Structural Summary (Rows, Columns, Memory, Duplicates)
- 📉 Missing Values Analysis (Table + Bar Chart)
- 📈 Basic Statistics Table
- 🔤 Data Types Summary
- 🔥 Correlation Matrix Heatmap
- 📊 High Correlation Scatter Plots (r ≥ 0.8)
- 📊 Distribution Histograms (all numerical columns)
- 📊 Categorical Bar Charts (all categorical columns)

**Endpoint:** `POST /run_complete_eda`

### **Train Models Tab** - Model Training ✅
Contains ONLY performance tables (NO GRAPHS):
- 🎯 Model Performance Table (LazyPredict-style)
- 🏆 Best Model Highlight
- 💾 Model Download Links

**Excluded:** ❌ Confusion Matrix, ❌ ROC Curve, ❌ Feature Importance

**Endpoint:** `POST /train_all_models`

---

## 🗂️ Complete Workflow

```
1. Upload Dataset
   POST /upload_dataset
   ↓
2. Clean Columns
   POST /advanced_cleanup
   └── Removes ID, zero-variance, constant columns
   ↓
3. Run Complete EDA (ALL VISUALIZATIONS HERE!)
   POST /run_complete_eda
   └── Structural summary, correlations, distributions
   ↓
4. Preprocess (Optional)
   POST /impute_missing_values
   POST /handle_outliers
   POST /preprocess
   ↓
5. Train Models (ONLY TABLE HERE!)
   POST /train_all_models
   └── Performance table with metrics
```

---

## 🛠️ Tech Stack

### Frontend
- **React** with TypeScript
- **Vite** for fast builds
- **TailwindCSS** for styling
- **shadcn/ui** for components
- **React Router** for navigation

### Backend
- **FastAPI** for REST API
- **Pandas** & **NumPy** for data processing
- **Scikit-learn** for ML models
- **XGBoost** & **LightGBM** for boosting
- **Plotly** for interactive visualizations
- **Matplotlib** & **Seaborn** for static plots

---

## 📚 API Documentation

Once backend is running, visit:
- **API Docs:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

### Key Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/upload_dataset` | POST | Upload CSV file |
| `/advanced_cleanup` | POST | Auto clean columns |
| `/run_complete_eda` | POST | Full EDA with visualizations |
| `/impute_missing_values` | POST | Handle missing values |
| `/handle_outliers` | POST | Handle outliers |
| `/train_all_models` | POST | Train all models |
| `/download_model/{name}` | GET | Download model file |

---

## 📖 Documentation

- **[API Update Guide](backend/API_UPDATE_GUIDE.md)** - Complete API documentation
- **[Implementation Guide](backend/IMPLEMENTATION_GUIDE.md)** - Technical details
- **[Strict Output Placement Demo](backend/STRICT_OUTPUT_PLACEMENT_DEMO.md)** - Flow verification

---

## 🎨 Features

### Automated Data Cleaning
- Removes ID/index columns automatically
- Detects and removes zero-variance columns
- Removes high-cardinality unique identifiers
- Duplicate row detection

### Comprehensive EDA
- **Structural Summary:** 5-metric overview table
- **Missing Values:** Markdown table + visualization
- **Correlation Analysis:** Heatmap + scatter plots (r ≥ 0.8)
- **Distribution Analysis:** Histograms for all numerical columns
- **Categorical Analysis:** Bar charts for categorical columns

### Model Training
- **79+ Models:** Comprehensive model library
- **Auto Problem Detection:** Classification vs Regression
- **Hyperparameter Tuning:** GridSearchCV/RandomizedSearchCV
- **Performance Table:** LazyPredict-style results
- **Model Export:** Download trained models (.joblib)

### Preprocessing
- **Missing Value Imputation:** Mean, Median, Mode, Forward/Backward Fill
- **Outlier Handling:** IQR, Z-Score, Winsorization, Percentile Clipping
- **Strategy Tracking:** Markdown table of applied techniques
- **Encoding & Scaling:** Label encoding + StandardScaler

---

## 🧪 Testing

### Test EDA Endpoint:
```bash
curl -X POST "http://localhost:8000/run_complete_eda?correlation_threshold=0.8"
```

**Expected:** JSON with correlation heatmap and distribution plots

### Test Train Endpoint:
```bash
curl -X POST "http://localhost:8000/train_all_models" \
  -H "Content-Type: application/json" \
  -d '{"target_column": "target", "with_tuning": false}'
```

**Expected:** JSON with `markdown_table` (NO plot URLs)

---
- Edit files directly within the Codespace and commit and push your changes once you're done.

## What technologies are used for this project?

This project is built with:

- Vite
- TypeScript
- React
- shadcn-ui
- Tailwind CSS

## How can I deploy this project?

Simply open [Lovable](https://lovable.dev/projects/41a3a9a7-f99a-4ea7-8916-a728c0cb53bb) and click on Share -> Publish.

## Can I connect a custom domain to my Lovable project?

Yes, you can!

To connect a domain, navigate to Project > Settings > Domains and click Connect Domain.

Read more here: [Setting up a custom domain](https://docs.lovable.dev/features/custom-domain#custom-domain)

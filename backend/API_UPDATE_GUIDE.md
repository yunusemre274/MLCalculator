# Makine Öğrenimi Web Uygulaması - API Güncelleme Dokümantasyonu

## 🎯 GÜNCELLENMİŞ GEREKSİNİMLER

### GEREKSİNİM 1: EDA Sekmesi İçin Kapsamlı ve Tablosal Özetler

#### A. Tablosal Veri Özeti (Structural Summary) ✅

**Endpoint:** `POST /run_complete_eda`

**Çıktı Örneği:**
```markdown
## 📊 Veri Seti Yapısal Özeti (Structural Summary)

| Metrik | Değer |
|--------|-------|
| Satır Sayısı (Rows) | 1,000 |
| Kolon Sayısı (Columns) | 15 |
| Toplam Hücre Sayısı (Total Cells) | 15,000 |
| Bellek Kullanımı (Memory Usage) | 1.25 MB |
| Tekrarlanan Satır Sayısı | 5 |
```

**JSON Response:**
```json
{
  "results": {
    "dataset_info": {
      "total_rows": 1000,
      "total_columns": 15,
      "total_cells": 15000,
      "memory_usage_mb": "1.25 MB",
      "duplicate_rows": 5,
      "markdown_table": "## 📊 Veri Seti Yapısal Özeti...",
      "column_names": ["col1", "col2", ...]
    }
  }
}
```

#### B. Korelasyon ve Dağılım Grafikleri ✅

**EDA sekmesi aşağıdaki tüm grafikleri içerir:**

1. **Veri Seti Yapısal Özeti** - Markdown tablo
2. **Eksik Değerler Analizi** - Markdown tablo + Bar grafik
3. **Temel İstatistikler** - Markdown tablo
4. **Veri Tipleri Özeti** - Markdown tablo
5. **Korelasyon Matrisi** - Plotly heatmap
6. **Yüksek Korelasyonlu Kolonlar** - Plotly scatter plots (r >= 0.8)
7. **Dağılım Grafikleri** - Plotly histograms (tüm numerik kolonlar için)
8. **Kategorik Grafikleri** - Plotly bar charts (tüm kategorik kolonlar için)

**Kullanım:**
```python
POST /run_complete_eda?correlation_threshold=0.8

Response:
{
  "message": "Complete EDA finished successfully",
  "results": {
    "dataset_info": { ... },
    "missing_values": { 
      "markdown_table": "...",
      "plot_html": "..."
    },
    "basic_statistics": { "markdown_table": "..." },
    "data_types_summary": { "markdown_table": "..." },
    "correlation_analysis": {
      "high_correlation_pairs": [
        {
          "column1": "age",
          "column2": "salary",
          "correlation": 0.85,
          "plot_html": "<plotly scatter plot>",
          "relationship": "Strong Positive"
        }
      ],
      "correlation_heatmap_html": "<plotly heatmap>"
    },
    "distribution_plots": {
      "age": "<plotly histogram>",
      "salary": "<plotly histogram>"
    },
    "categorical_plots": {
      "department": "<plotly bar chart>"
    }
  }
}
```

---

### GEREKSİNİM 2: Train Models Sekmesi İçin Sadece Tablosal Sonuçlar

#### A. Model Sonuçları Tablosu (LazyPredict Benzeri) ✅

**GRAFİKLER KALDIRILDI:** Confusion Matrix, ROC Curve, Feature Importance grafikleri artık döndürülmüyor.

**SADECE TABLO:** Model performansları LazyPredict benzeri Markdown tablosu olarak döndürülüyor.

**Endpoint:** `POST /train_all_models`

**Sınıflandırma Örneği:**
```markdown
## 🎯 Model Performans Özeti (Model Performance Summary)

| Model | Accuracy | F1 Score | Recall | Precision | Eğitim Süresi (Seconds) |
|-------|----------|----------|--------|-----------|-------------------------|
| XGBoost | 0.995 | 0.995 | 0.994 | 0.996 | 3.50 |
| Random Forest | 0.990 | 0.988 | 0.985 | 0.991 | 1.25 |
| Logistic Regression | 0.985 | 0.980 | 0.970 | 0.990 | 0.05 |
| GradientBoosting | 0.982 | 0.978 | 0.975 | 0.981 | 2.15 |
| SVC | 0.975 | 0.970 | 0.965 | 0.975 | 0.82 |

**🏆 En İyi Model (Best Model):** XGBoost (Accuracy: 0.995)
```

**Regresyon Örneği:**
```markdown
## 🎯 Model Performans Özeti (Model Performance Summary)

| Model | R² Score | RMSE | MAE | MSE | Eğitim Süresi (Seconds) |
|-------|----------|------|-----|-----|-------------------------|
| XGBoost | 0.985 | 12.35 | 8.45 | 152.52 | 3.25 |
| Random Forest | 0.978 | 15.20 | 10.25 | 231.04 | 1.50 |
| GradientBoosting | 0.972 | 17.85 | 12.10 | 318.62 | 2.80 |
| Linear Regression | 0.865 | 35.42 | 25.18 | 1254.58 | 0.02 |
| Ridge | 0.863 | 35.78 | 25.45 | 1280.21 | 0.03 |

**🏆 En İyi Model (Best Model):** XGBoost (R² Score: 0.985)
```

**JSON Response:**
```json
{
  "message": "All models trained successfully",
  "problem_type": "classification",
  "markdown_table": "## 🎯 Model Performans Özeti...",
  "results": [
    {
      "model_name": "XGBoost",
      "accuracy": 0.995,
      "f1_score": 0.995,
      "recall": 0.994,
      "precision": 0.996,
      "training_time": 3.50,
      "model_path": "/models/XGBoost.joblib"
    }
  ],
  "best_model": {
    "name": "XGBoost",
    "metric": "accuracy",
    "score": 0.995
  },
  "train_test_split": {
    "train_size": 800,
    "test_size": 200,
    "train_percentage": 80.0,
    "test_percentage": 20.0
  }
}
```

---

## 🔄 TAM İŞ AKIŞI

```
1. Upload Dataset
   POST /upload_dataset
   ↓
2. Clean Columns
   POST /advanced_cleanup
   ↓
3. Run Complete EDA (✅ GEREKSİNİM 1)
   POST /run_complete_eda
   └── Returns:
       ├── Structural Summary (Markdown table)
       ├── Missing Values (Markdown table + plot)
       ├── Basic Statistics (Markdown table)
       ├── Data Types Summary (Markdown table)
       ├── Correlation Matrix (Plotly heatmap)
       ├── High Correlation Plots (Plotly scatter, r >= 0.8)
       ├── Distribution Plots (Plotly histograms)
       └── Categorical Plots (Plotly bar charts)
   ↓
4. Preprocess (Optional)
   POST /impute_missing_values
   POST /handle_outliers
   POST /preprocess
   ↓
5. Train Models (✅ GEREKSİNİM 2)
   POST /train_all_models
   └── Returns:
       ├── Markdown table (LazyPredict style)
       ├── Best model info
       ├── Model metrics (NO GRAPHS!)
       └── Model file paths
```

---

## 📊 Frontend Entegrasyon Örnekleri

### EDA Sekmesi - Structural Summary Gösterimi

```typescript
// React component for EDA tab
const EDATab = () => {
  const [edaResults, setEdaResults] = useState(null);

  const runEDA = async () => {
    const response = await fetch('/run_complete_eda?correlation_threshold=0.8', {
      method: 'POST'
    });
    const data = await response.json();
    setEdaResults(data.results);
  };

  return (
    <div>
      <button onClick={runEDA}>Run EDA</button>
      
      {edaResults && (
        <>
          {/* Structural Summary */}
          <ReactMarkdown>
            {edaResults.dataset_info.markdown_table}
          </ReactMarkdown>
          
          {/* Missing Values */}
          <ReactMarkdown>
            {edaResults.missing_values.markdown_table}
          </ReactMarkdown>
          {edaResults.missing_values.plot_html && (
            <div dangerouslySetInnerHTML={{ 
              __html: edaResults.missing_values.plot_html 
            }} />
          )}
          
          {/* Correlation Heatmap */}
          <div dangerouslySetInnerHTML={{ 
            __html: edaResults.correlation_analysis.correlation_heatmap_html 
          }} />
          
          {/* High Correlation Scatter Plots */}
          {edaResults.correlation_analysis.high_correlation_pairs.map(pair => (
            <div key={`${pair.column1}_${pair.column2}`}>
              <h3>{pair.column1} vs {pair.column2} (r = {pair.correlation})</h3>
              <div dangerouslySetInnerHTML={{ __html: pair.plot_html }} />
            </div>
          ))}
          
          {/* Distribution Plots */}
          {Object.entries(edaResults.distribution_plots).map(([col, html]) => (
            <div key={col} dangerouslySetInnerHTML={{ __html: html }} />
          ))}
        </>
      )}
    </div>
  );
};
```

### Train Models Sekmesi - LazyPredict Tablo Gösterimi

```typescript
// React component for Train Models tab
const TrainTab = () => {
  const [trainResults, setTrainResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const trainModels = async () => {
    setLoading(true);
    const response = await fetch('/train_all_models', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        target_column: 'target',
        with_tuning: false
      })
    });
    const data = await response.json();
    setTrainResults(data);
    setLoading(false);
  };

  return (
    <div>
      <button onClick={trainModels} disabled={loading}>
        {loading ? 'Training...' : 'Train All Models'}
      </button>
      
      {trainResults && (
        <>
          {/* LazyPredict-style Table */}
          <ReactMarkdown>
            {trainResults.markdown_table}
          </ReactMarkdown>
          
          {/* Best Model Info */}
          <div className="best-model-card">
            <h3>🏆 En İyi Model</h3>
            <p>Model: {trainResults.best_model.name}</p>
            <p>Score: {trainResults.best_model.score}</p>
          </div>
          
          {/* Download Best Model */}
          <a 
            href={`/download_model/${trainResults.best_model.name}`}
            download
          >
            Download Best Model
          </a>
        </>
      )}
    </div>
  );
};
```

---

## ✅ ÖZETLER

### GEREKSİNİM 1: EDA Sekmesi ✅
- ✅ Tablosal Veri Özeti (5 metrik: Satır, Kolon, Hücre, Bellek, Tekrar)
- ✅ Tüm dağılım grafikleri (Plotly histograms)
- ✅ Korelasyon matrisi (Plotly heatmap)
- ✅ Koşullu scatter plotlar (r >= 0.8)
- ✅ Eksik değerler Markdown tablo
- ✅ Temel istatistikler Markdown tablo
- ✅ Veri tipleri Markdown tablo

### GEREKSİNİM 2: Train Models Sekmesi ✅
- ✅ Grafikler KALDIRILDI (Confusion Matrix, ROC, Feature Importance)
- ✅ LazyPredict benzeri Markdown tablo
- ✅ Sınıflandırma: Accuracy, F1, Recall, Precision, Eğitim Süresi
- ✅ Regresyon: R², RMSE, MAE, MSE, Eğitim Süresi
- ✅ Modeller performansa göre sıralanmış
- ✅ En iyi model vurgulanmış

---

## 🚀 Test Adımları

1. Backend'i başlat:
```bash
cd backend
C:/Users/yunus/Desktop/Projects/MachineLearningCalculator/.venv/Scripts/python.exe -m uvicorn app.main:app --reload
```

2. Frontend'i başlat:
```bash
npm run dev
```

3. Test et:
- Upload bir dataset
- Run EDA → Structural Summary tablosunu gör
- Run EDA → Korelasyon ve dağılım grafiklerini gör
- Train Models → LazyPredict tablosunu gör (grafik YOK!)

Tüm gereksinimler hazır! 🎉

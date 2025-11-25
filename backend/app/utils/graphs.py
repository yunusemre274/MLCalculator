import matplotlib.pyplot as plt
import seaborn as sns
import os
import uuid
from .file_manager import FileManager

# Set non-interactive backend
plt.switch_backend('Agg')

class GraphGenerator:
    @staticmethod
    def _save_plot(filename_prefix: str) -> str:
        filename = f"{filename_prefix}_{uuid.uuid4().hex[:8]}.png"
        filepath = FileManager.get_image_path(filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        return FileManager.get_relative_image_path(filename)

    @staticmethod
    def plot_histogram(df, column):
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], kde=True)
        plt.title(f'Distribution of {column}')
        return GraphGenerator._save_plot(f"hist_{column}")

    @staticmethod
    def plot_correlation_matrix(df):
        plt.figure(figsize=(12, 10))
        # Select only numeric columns for correlation matrix
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        if numeric_df.empty:
            return None
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        return GraphGenerator._save_plot("correlation_matrix")

    @staticmethod
    def plot_countplot(df, column):
        plt.figure(figsize=(10, 6))
        value_counts = df[column].value_counts()
        if len(value_counts) > 20:
            value_counts = value_counts.head(20)
        sns.countplot(y=df[df[column].isin(value_counts.index)][column], order=value_counts.index)
        plt.title(f'Count Plot of {column} (Top 20)')
        plt.xlabel('Count')
        return GraphGenerator._save_plot(f"count_{column}")
    
    @staticmethod
    def plot_pairplot(df, columns):
        """Generate pairplot for selected numerical columns"""
        try:
            pairplot_fig = sns.pairplot(df[columns], diag_kind='kde', plot_kws={'alpha': 0.6})
            pairplot_fig.fig.suptitle('Pairplot of Numerical Features', y=1.01)
            filename = f"pairplot_{uuid.uuid4().hex[:8]}.png"
            filepath = FileManager.get_image_path(filename)
            pairplot_fig.savefig(filepath, bbox_inches='tight', dpi=100)
            plt.close()
            return FileManager.get_relative_image_path(filename)
        except:
            return None
    
    @staticmethod
    def plot_kde(df, column):
        """Generate KDE plot"""
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df[column], fill=True)
        plt.title(f'KDE Plot of {column}')
        plt.xlabel(column)
        return GraphGenerator._save_plot(f"kde_{column}")
    
    @staticmethod
    def plot_scatter(df, col1, col2, correlation):
        """Generate scatter plot for two correlated columns"""
        plt.figure(figsize=(10, 6))
        plt.scatter(df[col1], df[col2], alpha=0.5)
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.title(f'Scatter Plot: {col1} vs {col2} (Correlation: {correlation:.3f})')
        
        # Add trend line
        z = np.polyfit(df[col1].dropna(), df[col2].dropna(), 1)
        p = np.poly1d(z)
        plt.plot(df[col1], p(df[col1]), "r--", alpha=0.8, label='Trend Line')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return GraphGenerator._save_plot(f"scatter_{col1}_{col2}")

    @staticmethod
    def plot_confusion_matrix(cm, classes):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        return GraphGenerator._save_plot("confusion_matrix")

    @staticmethod
    def plot_roc_curve(fpr, tpr, auc_score):
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        return GraphGenerator._save_plot("roc_curve")

    @staticmethod
    def plot_feature_importance(importances, feature_names):
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances, y=feature_names)
        plt.title('Feature Importances')
        return GraphGenerator._save_plot("feature_importance")

    @staticmethod
    def plot_residuals(y_test, y_pred):
        plt.figure(figsize=(10, 6))
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        return GraphGenerator._save_plot("residuals")

    @staticmethod
    def plot_predicted_vs_actual(y_test, y_pred):
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Predicted vs Actual')
        return GraphGenerator._save_plot("pred_vs_actual")

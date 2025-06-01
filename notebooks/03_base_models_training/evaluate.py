import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.metrics import r2_score, root_mean_squared_error
from pandas import DataFrame, Series

Numeric = Union[np.ndarray, Series]

def compute_metrics(
    y_true: Numeric,
    y_pred: Numeric
) -> Tuple[float, float]:
    """
    Вычисляет коэффициент детерминации R² и RMSE.
    Возвращает кортеж (r2, rmse).
    """
    r2 = r2_score(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    return r2, rmse


def plot_actual_vs_predicted(
    y_train: Numeric,
    y_pred_train: Numeric,
    y_test: Numeric,
    y_pred_test: Numeric,
    model_name: str,
    metrics_train: Tuple[float, float],
    metrics_test: Tuple[float, float]
) -> None:
    """
    Строит scatterplot: фактические и предсказанные значения для обучающей и тестовой выборок.
    Метрики R² и RMSE включаются в легенду.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=y_train, y=y_pred_train,
        label=(f"Обучение: R²={metrics_train[0]:.3f}, RMSE={metrics_train[1]:.3f}"),
        marker='o', alpha=0.6
    )
    sns.scatterplot(
        x=y_test, y=y_pred_test,
        label=(f"Тест: R²={metrics_test[0]:.3f}, RMSE={metrics_test[1]:.3f}"),
        marker='X', alpha=0.8
    )

    # Диагональ y = x
    lo = min(y_train.min(), y_test.min())
    hi = max(y_train.max(), y_test.max())
    plt.plot([lo, hi], [lo, hi], 'k--', lw=1.2)

    plt.title(f"{model_name}\nФактические и Предсказанные")
    plt.xlabel("Фактические значения")
    plt.ylabel("Предсказанные значения")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def evaluate_holdout(
    X_train: DataFrame,
    X_test: DataFrame,
    y_train: Numeric,
    y_test: Numeric,
    model: Any,
    model_name: str,
    **kwargs: Dict[str, Any]
) -> Dict[str, float]:
    """
    Полная оценка модели на разбиении train/test:
      1. Обучение модели
      2. Предсказания на train и test
      3. Расчет метрик R² и RMSE
      4. Визуализация результатов
    Возвращает словарь с метриками.
    """
    # 1. Обучение
    trained_model = model.fit(X_train, y_train, **kwargs)

    # 2. Предсказания
    y_pred_train = trained_model.predict(X_train)
    y_pred_test  = trained_model.predict(X_test)

    # 3. Метрики
    metrics_train = compute_metrics(y_train, y_pred_train)
    metrics_test  = compute_metrics(y_test,  y_pred_test)

    # 4. График
    plot_actual_vs_predicted(
        y_train, y_pred_train,
        y_test, y_pred_test,
        model_name,
        metrics_train,
        metrics_test
    )

    return {
        "r2_train":  metrics_train[0],
        "rmse_train": metrics_train[1],
        "r2_test":   metrics_test[0],
        "rmse_test":  metrics_test[1]
    }

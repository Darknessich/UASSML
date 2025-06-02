# Data Augmentation Module

""" 
Этот модуль предоставляет функции для синтетического расширения набора данных 
путём добавления случайного шума к числовым полям, корректировки зависимых 
значений и визуализации результатов.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 24})

## 1. Генерация случайного отклонения
def vary(value: float, min_pct: float = 0.02, max_pct: float = 0.05) -> float:
    """
    Добавляет к числовому значению value случайное отклонение в диапазоне
    [-value * max_pct, +value * max_pct], причём уровень шума
    варьируется между min_pct и max_pct.

    :param value: исходное числовое значение
    :param min_pct: минимальная доля отклонения (по умолчанию 2%)
    :param max_pct: максимальная доля отклонения (по умолчанию 5%)
    :return: новое значение с учётом стохастического шума
    """
    noise_level = np.random.uniform(min_pct, max_pct)
    return value + np.random.uniform(-noise_level, noise_level) * value


## 2. Аугментация одной строки
def augment_row(row: dict, schema: dict) -> dict:
    """
    Создаёт аугментированную копию словаря row по правилу, описанному в schema.

    :param row: словарь {имя_поля: значение}
    :param schema: словарь {имя_поля: (min_pct, max_pct)}
                   указывает диапазоны шума для выбранных полей
    :return: новый словарь с модифицированными значениями
    """
    new_row = row.copy()
    for field, (min_pct, max_pct) in schema.items():
        try:
            num = float(row[field])
            new_row[field] = round(vary(num, min_pct, max_pct), 3)
        except (TypeError, ValueError, KeyError):
            # Пропускаем, если поле отсутствует или нечисловое
            pass
    return new_row


## 3. Аугментация всего DataFrame
def augment_dataframe(df: pd.DataFrame, schema: dict, n: int = 4) -> pd.DataFrame:
    """
    Для каждой строки исходного df создаёт n аугментированных копий
    и объединяет их с оригиналом.

    :param df: исходный DataFrame
    :param schema: схема аугментации для числовых полей
    :param n: количество копий для каждой строки (по умолчанию 4)
    :return: новый DataFrame с расширенным набором строк
    """
    rows = []
    for _, row in df.iterrows():
        base_row = row.to_dict()
        rows.append(base_row)
        for _ in range(n):
            rows.append(augment_row(base_row, schema))
    return pd.DataFrame(rows)


## 4. Исправление несоответствий value/max
def fix_max_ge_value(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """
    Убедиться, что для всех пар колонок '<prefix>.value' и '<prefix>.max'
    значение max не ниже value.

    :param df: DataFrame с результатами экспериментов
    :param schema: схема, содержащая возможные пары колонок
    :return: скорректированный DataFrame
    """
    value_cols = [col for col in schema if col.endswith('.value')]
    max_cols = [col for col in schema if col.endswith('.max')]

    for v in value_cols:
        prefix = v[:-6]  # удаляем '.value'
        m = prefix + '.max'
        if m in df.columns:
            mask = df[m] < df[v]
            df.loc[mask, m] = df.loc[mask, v]
    return df


## 5. Визуализация метрик в чёрно-белом стиле
def plot_metric_bw_style(
    df_orig: pd.DataFrame,
    df_aug5: pd.DataFrame,
    df_aug10: pd.DataFrame,
    model_name: str,
    metric: str,
    experiment_order: list = None,
    y_value_range: tuple = None,
    y_label: str = None
) -> None:
    """
    Строит столбчатые графики средних значений метрики
    для оригинальных данных и аугментаций x5, x10.

    :param df_orig: DataFrame оригинальных данных
    :param df_aug5: DataFrame с 5-кратной аугментацией
    :param df_aug10: DataFrame с 10-кратной аугментацией
    :param model_name: имя модели для фильтрации
    :param metric: имя метрики (например, 'coverage' или 'droplet_size')
    :param experiment_order: список имен экспериментов для упорядочивания
    :param y_value_range: кортеж (min, max) для оси значений
    """
    value_col = f"experiment.results.{metric}.value"

    def avg(df: pd.DataFrame) -> pd.DataFrame:
        sub = df[df['model.name'] == model_name]
        return sub.groupby('experiment.name')[[value_col]].mean().reset_index()

    df_o = avg(df_orig)
    df_5 = avg(df_aug5)
    df_10 = avg(df_aug10)

    common = sorted(set(df_o['experiment.name']) & set(df_5['experiment.name']) & set(df_10['experiment.name']))
    if experiment_order:
        common = [e for e in experiment_order if e in common]

    def prepare(df: pd.DataFrame) -> pd.DataFrame:
        df = df[df['experiment.name'].isin(common)].copy()
        df['experiment.name'] = pd.Categorical(df['experiment.name'], categories=common, ordered=True)
        return df.sort_values('experiment.name')

    df_o, df_5, df_10 = map(prepare, (df_o, df_5, df_10))
    x = range(len(common))

    fig, ax1 = plt.subplots(figsize=(14, 7))

    width = 0.25
    ax1.bar([i - width for i in x], df_o[value_col], width,
             color='white', edgecolor='black', hatch='///', label=f"{metric} value (orig)")
    ax1.bar(x, df_5[value_col], width, 
            color='lightgrey', edgecolor='black', hatch='\\\\\\', label=f"{metric} value (aug x5)")
    ax1.bar([i + width for i in x], df_10[value_col], width, 
            color='grey', edgecolor='black', hatch='xxx', label=f"{metric} value (aug x10)")

    if y_value_range: ax1.set_ylim(y_value_range)

    ax1.set_xticks(x)
    ax1.set_xticklabels(common, rotation=45, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.set_ylabel(y_label or f"{metric} value")

    ax1.legend(loc='upper left', frameon=False)

    plt.title(f"{model_name} — {metric}")
    plt.show()

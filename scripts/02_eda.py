"""
02_eda.py
探索性数据分析 (Exploratory Data Analysis)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "doctors_sample.csv"
FIGURES_PATH = PROJECT_ROOT / "figures"

def run_eda():
    """执行探索性数据分析"""
    
    # 确保图表目录存在
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    
    print("加载数据...")
    df = pd.read_csv(DATA_PATH)
    
    print(f"\n=== 数据集基本信息 ===")
    print(f"样本数量: {len(df)}")
    print(f"特征数量: {len(df.columns)}")
    print(f"\n字段列表:\n{df.dtypes}")
    
    # ==================== 1. 缺失值分析 ====================
    print("\n=== 缺失值分析 ===")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'缺失数量': missing, '缺失率(%)': missing_pct})
    print(missing_df[missing_df['缺失数量'] > 0])
    
    # ==================== 2. RFM 特征统计 ====================
    print("\n=== RFM 特征统计描述 ===")
    rfm_cols = ['rfm_monetary', 'rfm_frequency', 'recency_days']
    print(df[rfm_cols].describe())
    
    # ==================== 3. RFM 分布图 ====================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Monetary 分布 (使用对数变换)
    axes[0].hist(np.log1p(df['rfm_monetary']), bins=50, color='steelblue', edgecolor='white')
    axes[0].set_title('Monetary 分布 (log1p 变换)', fontsize=12)
    axes[0].set_xlabel('log(1 + Monetary)')
    axes[0].set_ylabel('频数')
    
    # Frequency 分布
    axes[1].hist(np.log1p(df['rfm_frequency']), bins=50, color='darkorange', edgecolor='white')
    axes[1].set_title('Frequency 分布 (log1p 变换)', fontsize=12)
    axes[1].set_xlabel('log(1 + Frequency)')
    axes[1].set_ylabel('频数')
    
    # Recency 分布
    axes[2].hist(df['recency_days'], bins=50, color='seagreen', edgecolor='white')
    axes[2].set_title('Recency 分布 (天数)', fontsize=12)
    axes[2].set_xlabel('距今天数')
    axes[2].set_ylabel('频数')
    
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / 'rfm_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {FIGURES_PATH / 'rfm_distribution.png'}")
    
    # ==================== 4. 相关性热力图 ====================
    fig, ax = plt.subplots(figsize=(8, 6))
    corr_matrix = df[rfm_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title('RFM 特征相关性矩阵', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {FIGURES_PATH / 'correlation_heatmap.png'}")
    
    # ==================== 5. 专业领域分布 (Top 10) ====================
    if 'specialty' in df.columns and df['specialty'].notna().any():
        fig, ax = plt.subplots(figsize=(12, 6))
        top_specialties = df['specialty'].value_counts().head(10)
        top_specialties.plot(kind='barh', color='teal', ax=ax)
        ax.set_title('Top 10 专业领域分布', fontsize=14)
        ax.set_xlabel('医生数量')
        ax.set_ylabel('专业领域')
        ax.invert_yaxis()  # 最多的在上面
        plt.tight_layout()
        plt.savefig(FIGURES_PATH / 'specialty_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"图表已保存: {FIGURES_PATH / 'specialty_distribution.png'}")
    
    # ==================== 6. 现有聚类分布 ====================
    if 'cluster_label' in df.columns and df['cluster_label'].notna().any():
        fig, ax = plt.subplots(figsize=(8, 5))
        cluster_counts = df['cluster_label'].value_counts()
        cluster_counts.plot(kind='bar', color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'], ax=ax)
        ax.set_title('现有聚类分布', fontsize=14)
        ax.set_xlabel('客户群体')
        ax.set_ylabel('医生数量')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(FIGURES_PATH / 'cluster_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"图表已保存: {FIGURES_PATH / 'cluster_distribution.png'}")
    
    print("\n=== EDA 完成 ===")
    print(f"所有图表已保存至: {FIGURES_PATH}")

if __name__ == "__main__":
    run_eda()

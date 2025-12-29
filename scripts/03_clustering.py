"""
03_clustering.py
K-Means 聚类分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "doctors_sample.csv"
FIGURES_PATH = PROJECT_ROOT / "figures"
OUTPUT_PATH = PROJECT_ROOT / "data" / "doctors_clustered.csv"

def run_clustering():
    """执行 K-Means 聚类分析"""
    
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    
    print("加载数据...")
    df = pd.read_csv(DATA_PATH)
    
    # RFM 特征
    rfm_cols = ['rfm_monetary', 'rfm_frequency', 'recency_days']
    
    # 移除缺失值
    df_clean = df.dropna(subset=rfm_cols).copy()
    print(f"有效样本数: {len(df_clean)}")
    
    # ==================== 1. 数据标准化 ====================
    print("\n=== 数据标准化 ===")
    
    # 对 Monetary 和 Frequency 进行 log 变换（减少长尾影响）
    df_clean['log_monetary'] = np.log1p(df_clean['rfm_monetary'])
    df_clean['log_frequency'] = np.log1p(df_clean['rfm_frequency'])
    
    # 选择用于聚类的特征
    features = ['log_monetary', 'log_frequency', 'recency_days']
    X = df_clean[features].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"特征: {features}")
    print(f"标准化后均值: {X_scaled.mean(axis=0).round(4)}")
    print(f"标准化后标准差: {X_scaled.std(axis=0).round(4)}")
    
    # ==================== 2. 肘部法则确定 K 值 ====================
    print("\n=== 肘部法则分析 ===")
    
    k_range = range(2, 11)
    inertias = []
    silhouettes = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))
        print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouettes[-1]:.4f}")
    
    # 绘制肘部法则图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 肘部法则
    axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('聚类数量 K', fontsize=12)
    axes[0].set_ylabel('惯性 (Inertia)', fontsize=12)
    axes[0].set_title('肘部法则 (Elbow Method)', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # 轮廓系数
    axes[1].plot(k_range, silhouettes, 'rs-', linewidth=2, markersize=8)
    axes[1].set_xlabel('聚类数量 K', fontsize=12)
    axes[1].set_ylabel('轮廓系数 (Silhouette Score)', fontsize=12)
    axes[1].set_title('轮廓系数曲线', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / 'elbow_silhouette.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {FIGURES_PATH / 'elbow_silhouette.png'}")
    
    # ==================== 3. 执行 K-Means (K=3) ====================
    optimal_k = 3  # 根据肘部法则和业务场景选择
    print(f"\n=== 执行 K-Means 聚类 (K={optimal_k}) ===")
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df_clean['new_cluster_id'] = kmeans.fit_predict(X_scaled)
    
    final_silhouette = silhouette_score(X_scaled, df_clean['new_cluster_id'])
    print(f"最终轮廓系数: {final_silhouette:.4f}")
    
    # ==================== 4. 各群体 RFM 特征分析 ====================
    print("\n=== 各群体 RFM 特征 ===")
    
    cluster_stats = df_clean.groupby('new_cluster_id').agg({
        'rfm_monetary': ['mean', 'median', 'count'],
        'rfm_frequency': ['mean', 'median'],
        'recency_days': ['mean', 'median']
    }).round(2)
    
    print(cluster_stats)
    
    # 根据特征命名群体
    cluster_means = df_clean.groupby('new_cluster_id')[['rfm_monetary', 'rfm_frequency', 'recency_days']].mean()
    
    # 按 Monetary 排序，高价值群排第一
    sorted_clusters = cluster_means.sort_values('rfm_monetary', ascending=False)
    
    label_map = {}
    labels = ['高价值核心群', '中价值潜力群', '低价值长尾群']
    for i, cluster_id in enumerate(sorted_clusters.index):
        label_map[cluster_id] = labels[i] if i < len(labels) else f'群体{cluster_id}'
    
    df_clean['new_cluster_label'] = df_clean['new_cluster_id'].map(label_map)
    
    print("\n=== 群体标签映射 ===")
    for cid, label in label_map.items():
        count = (df_clean['new_cluster_id'] == cid).sum()
        pct = count / len(df_clean) * 100
        print(f"Cluster {cid} -> {label} ({count} 人, {pct:.1f}%)")
    
    # ==================== 5. 可视化 ====================
    # 3D 散点图
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    for cid in sorted(df_clean['new_cluster_id'].unique()):
        mask = df_clean['new_cluster_id'] == cid
        ax.scatter(
            df_clean.loc[mask, 'log_monetary'],
            df_clean.loc[mask, 'log_frequency'],
            df_clean.loc[mask, 'recency_days'],
            c=colors[cid % len(colors)],
            label=label_map[cid],
            alpha=0.6,
            s=20
        )
    
    ax.set_xlabel('log(Monetary)')
    ax.set_ylabel('log(Frequency)')
    ax.set_zlabel('Recency (天)')
    ax.set_title('K-Means 聚类结果 (3D)', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / 'cluster_3d.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {FIGURES_PATH / 'cluster_3d.png'}")
    
    # 群体分布柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    cluster_counts = df_clean['new_cluster_label'].value_counts()
    bars = ax.bar(cluster_counts.index, cluster_counts.values, 
                  color=['#e74c3c', '#3498db', '#2ecc71'])
    ax.set_ylabel('医生数量', fontsize=12)
    ax.set_title('聚类结果 - 各群体人数分布', fontsize=14)
    
    # 添加数值标签
    for bar, count in zip(bars, cluster_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                f'{count}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / 'cluster_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {FIGURES_PATH / 'cluster_bar.png'}")
    
    # ==================== 6. 保存结果 ====================
    df_clean.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
    print(f"\n聚类结果已保存: {OUTPUT_PATH}")
    
    print("\n=== 聚类分析完成 ===")
    
    return df_clean, kmeans, scaler

if __name__ == "__main__":
    run_clustering()

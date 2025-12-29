"""
04_random_forest.py
随机森林分类 - 预测高价值客户
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "doctors_clustered.csv"
FIGURES_PATH = PROJECT_ROOT / "figures"

def run_random_forest():
    """执行随机森林分类"""
    
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    
    print("加载聚类后的数据...")
    df = pd.read_csv(DATA_PATH)
    
    # ==================== 1. 构造目标变量 ====================
    print("\n=== 构造目标变量 ===")
    
    # 将"高价值核心群"标记为 1 (正类)，其他为 0
    df['is_high_value'] = (df['new_cluster_label'] == '高价值核心群').astype(int)
    
    print(f"目标变量分布:")
    print(df['is_high_value'].value_counts())
    print(f"正类比例: {df['is_high_value'].mean():.2%}")
    
    # ==================== 2. 特征工程 ====================
    print("\n=== 特征工程 ===")
    
    # 选择特征
    feature_cols = ['rfm_monetary', 'rfm_frequency', 'recency_days']
    
    # 添加衍生特征
    df['monetary_per_freq'] = df['rfm_monetary'] / (df['rfm_frequency'] + 1)  # 单次交易金额
    df['activity_score'] = df['rfm_frequency'] / (df['recency_days'] + 1) * 100  # 活跃度指标
    
    feature_cols.extend(['monetary_per_freq', 'activity_score'])
    
    print(f"使用的特征: {feature_cols}")
    
    X = df[feature_cols].fillna(0)
    y = df['is_high_value']
    
    # ==================== 3. 划分训练/测试集 ====================
    print("\n=== 划分数据集 ===")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"训练集大小: {len(X_train)} ({len(X_train)/len(X):.1%})")
    print(f"测试集大小: {len(X_test)} ({len(X_test)/len(X):.1%})")
    
    # ==================== 4. 训练随机森林 ====================
    print("\n=== 训练随机森林模型 ===")
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    print("模型训练完成!")
    
    # ==================== 5. 模型评估 ====================
    print("\n=== 模型评估 ===")
    
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    
    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1 分数: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['非高价值', '高价值']))
    
    # ==================== 6. 混淆矩阵 ====================
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['非高价值', '高价值'],
                yticklabels=['非高价值', '高价值'], ax=ax)
    ax.set_xlabel('预测值', fontsize=12)
    ax.set_ylabel('真实值', fontsize=12)
    ax.set_title('混淆矩阵', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {FIGURES_PATH / 'confusion_matrix.png'}")
    
    # ==================== 7. ROC 曲线 ====================
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'Random Forest (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='随机猜测')
    ax.set_xlabel('假阳性率 (FPR)', fontsize=12)
    ax.set_ylabel('真阳性率 (TPR)', fontsize=12)
    ax.set_title('ROC 曲线', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / 'roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {FIGURES_PATH / 'roc_curve.png'}")
    
    # ==================== 8. 特征重要性 (核心产出) ====================
    print("\n=== 特征重要性排序 ===")
    
    feature_importance = pd.DataFrame({
        '特征': feature_cols,
        '重要性': rf.feature_importances_
    }).sort_values('重要性', ascending=False)
    
    print(feature_importance.to_string(index=False))
    
    # 绘制特征重要性图
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(feature_importance['特征'], feature_importance['重要性'], 
                   color='steelblue')
    ax.set_xlabel('特征重要性', fontsize=12)
    ax.set_ylabel('特征', fontsize=12)
    ax.set_title('随机森林 - 特征重要性排序', fontsize=14)
    ax.invert_yaxis()  # 最重要的在上面
    
    # 添加数值标签
    for bar, val in zip(bars, feature_importance['重要性']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {FIGURES_PATH / 'feature_importance.png'}")
    
    # ==================== 9. 保存结果摘要 ====================
    summary = f"""
=== 随机森林分类结果摘要 ===

数据集规模: {len(df)} 条记录
训练集: {len(X_train)} | 测试集: {len(X_test)}

模型参数:
- n_estimators: 100
- max_depth: 10
- min_samples_split: 10

模型性能:
- Accuracy: {accuracy:.4f}
- Precision: {precision:.4f}
- Recall: {recall:.4f}
- F1 Score: {f1:.4f}
- AUC: {auc:.4f}

特征重要性排序:
{feature_importance.to_string(index=False)}

结论: 模型能够有效识别高价值客户，AUC 达到 {auc:.2f}，
其中最重要的预测因素是 {feature_importance.iloc[0]['特征']}。
"""
    
    summary_path = PROJECT_ROOT / "data" / "rf_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"\n结果摘要已保存: {summary_path}")
    
    print("\n=== 随机森林分析完成 ===")
    
    return rf, feature_importance

if __name__ == "__main__":
    run_random_forest()

"""
01_data_export.py
从 pharma.db 导出医生数据用于分析
"""

import sqlite3
import pandas as pd
from pathlib import Path

# 路径配置
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT.parent / "backend" / "pharma.db"
OUTPUT_PATH = PROJECT_ROOT / "data" / "doctors_sample.csv"

def export_data(sample_size: int = None):
    """
    从数据库导出医生数据
    
    Args:
        sample_size: 抽样数量，None 表示全量导出
    """
    print(f"连接数据库: {DB_PATH}")
    
    # 确保输出目录存在
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    
    # 查询核心字段
    query = """
    SELECT 
        npi,
        full_name,
        specialty,
        state,
        city,
        monetary as rfm_monetary,
        frequency as rfm_frequency,
        recency_days,
        cluster_id,
        cluster_label
    FROM doctors
    WHERE monetary IS NOT NULL 
      AND frequency IS NOT NULL
    """
    
    if sample_size:
        query += f" ORDER BY RANDOM() LIMIT {sample_size}"
    
    print("执行查询...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"导出记录数: {len(df)}")
    print(f"字段: {list(df.columns)}")
    
    # 保存为 CSV
    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
    print(f"数据已保存至: {OUTPUT_PATH}")
    
    # 打印基本统计信息
    print("\n=== 数据概览 ===")
    print(df.describe())
    
    return df

if __name__ == "__main__":
    # 导出 10000 条抽样数据（加快后续分析速度）
    # 如需全量数据，设置 sample_size=None
    export_data(sample_size=10000)

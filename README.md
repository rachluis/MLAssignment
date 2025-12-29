# ML/DL 课程作业 - 医药行业KOL价值评估

## 项目简介

本项目基于 CMS Open Payments 医药支付数据，使用 K-Means 聚类 + 随机森林分类的方法，对医生（KOL）进行价值分层与归因分析。

## 数据来源

- **数据库**: `../backend/pharma.db`
- **核心表**: `doctors` (738,772 条记录)
- **特征**: RFM 模型 (Recency, Frequency, Monetary)

## 项目结构

```
MLAssignment/
├── data/                       # 导出的数据文件
├── notebooks/                  # Jupyter Notebook
├── scripts/                    # Python 脚本
│   ├── 01_data_export.py       # 数据导出
│   ├── 02_eda.py               # 探索性数据分析
│   ├── 03_clustering.py        # K-Means 聚类
│   └── 04_random_forest.py     # 随机森林分类
├── figures/                    # 生成的图表
└── report/                     # 论文
    └── paper.md
```

## 快速开始

```bash
# 1. 进入项目目录
cd MLAssignment

# 2. 导出数据
python scripts/01_data_export.py

# 3. 运行探索性分析
python scripts/02_eda.py

# 4. 运行聚类分析
python scripts/03_clustering.py

# 5. 运行随机森林分类
python scripts/04_random_forest.py
```

## 依赖

- Python 3.9+
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- sqlite3 (内置)

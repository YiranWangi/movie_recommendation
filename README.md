# 推荐系统项目

这是一个基于 MovieLens 100K 数据集的推荐系统项目，包含模型训练、推理、API 服务以及可视化功能。

## 目录结构
.
├── api.py            # 启动推荐 API 服务
├── train.py          # 模型训练脚本
├── inference.py      # 模型推理脚本
├── utils.py          # 工具函数（数据处理、评估等）
├── visualize.py      # 可视化脚本
├── data/             # 数据目录
│   ├── u.data        # 用户评分数据
│   ├── u.item        # 电影信息数据
│   ├── u.user        # 用户信息数据
│   └── u.genre       # 电影类型映射
└── README.md         # 项目说明文件

## 环境依赖
- Python >= 3.7
- 推荐使用 virtualenv 或 conda 管理虚拟环境

## 数据集
本项目使用 MovieLens 100K 数据集，包含以下文件：
- u.data: 用户对电影的评分，格式为 user_id movie_id rating timestamp
- u.item: 电影信息，包含 movie_id | movie_title | release_date |
- u.user: 用户信息，包含 user_id | age | gender | occupation | zip_code
- u.genre: 电影类型映射，每行 genre_name|genre_id

## 使用说明

### 模型训练
```bash
python train.py --data_dir data --epochs 20 --batch_size 128 --lr 0.001 --output_dir checkpoints
```

### 模型推理
```bash
python inference.py --model_path checkpoints/model.pth --data_dir data --output predictions.csv
```

### API服务
```bash
python api.py --host 0.0.0.0 --port 5000 --model_path checkpoints/model.pth --data_dir data
```

### 可视化
```bash
python visualize.py --predictions predictions.csv --output_dir results/figures
```

## 代码说明
- utils.py：包含数据加载、预处理、评估指标等通用函数
- train.py：负责数据读取、模型构建、训练及日志记录
- inference.py：负责加载模型并生成评分预测
- api.py：基于 Flask 提供在线推荐服务
- visualize.py：负责读取推理结果并生成可视化图表

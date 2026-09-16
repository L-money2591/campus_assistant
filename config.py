import os
# 大模型API配置（优先从环境变量读取）
API_KEY = os.getenv("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")

# Embedding向量模型
EMBEDDING_MODEL_NAME = "shibing624/text2vec-base-chinese"

# 向量库存放路径
DB_PATH = "./vector_db"

# LRU缓存最大条数
CACHE_MAX = 50

# ===================== LoRA微调配置 =====================
# 修复：train_data.json 在仓库根目录，默认路径改为根路径或调整到 data/ 目录
TRAIN_DATA_PATH = "./train_data.json"
LORA_WEIGHT_PATH = "./lora_weights/"
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
MAX_TRAIN_EPOCH = 8
TRAIN_BATCH_SIZE = 4
# False=在线API+提示词模拟微调；True=本地加载LoRA微调模型
USE_FINETUNE_MODEL = False

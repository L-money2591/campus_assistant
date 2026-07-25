校园智能助手
基于 FastAPI + LangChain + RAG + Agent + LoRA 微调 的校园问答系统，提供自然语言交互的校园信息服务，支持多种问答模式，并可选加载本地微调模型。

✨ 功能特性

多种问答模式：
normal / double：双 Agent 协作（检索 + 口语化生成），模拟微调风格
route：内置 CoT 思维链，专为行程路线规划设计
tool：Agent 工具模式，可查询空教室、天气等

RAG 增强检索：基于本地向量库，精准匹配校园知识（图书馆、食堂、快递等）
LRU 缓存：提高重复问题响应速度
美观 Web 界面：毛玻璃效果，响应式设计，支持缓存标记
灵活 LLM 后端：支持在线 API（如 DeepSeek）或本地加载 LoRA 微调模型
一键微调：提供 LoRA 微调脚本，可基于自定义数据训练专属风格

📁 项目结构
.
├── app.py                  # FastAPI 主应用，定义 /ask 接口与前端路由
├── build_db.py             # 向量数据库构建脚本
├── config.py               # 全局配置（API、路径、微调参数等）
├── train_lora.py           # LoRA 微调训练脚本
├── train_data.json         # 微调训练数据集（指令-回答对）
├── requirements.txt        # Python 依赖列表
├── templates/
│   └── index.html          # 前端聊天页面
├── 图书馆.txt              # 图书馆相关文本知识
├── 食堂快递.txt            # 食堂、快递相关文本知识
└── README.md               # 本文档

🚀 快速开始
1. 环境准备
Python 版本：≥ 3.9

安装依赖：
pip install -r requirements.txt

2. 配置
编辑 config.py 或通过环境变量设置：

# 推荐使用环境变量（安全）
API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-api-key")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")

# 本地向量库路径
DB_PATH = "./vector_db"

# 是否启用本地微调模型（需先训练）
USE_FINETUNE_MODEL = False

3. 构建向量数据库
运行以下命令，将 图书馆.txt 和 食堂快递.txt 分块并存入 Chroma 向量库：
python build_db.py
成功后会生成 ./vector_db 目录。

4. 启动服务
python app.py
服务将在 http://0.0.0.0:8000 启动，浏览器访问即可打开聊天界面。

🧠 使用说明
Web 界面
在输入框键入问题，按回车或点击“发送”。
顶部下拉菜单选择问答模式：

普通问答：默认模式，使用双 Agent（检索+生成）
路线规划：启用 CoT 思维链，适合“几点下课去图书馆还书再去食堂”类问题
工具查询：调用 Agent 工具（如 empty_classroom、campus_weather）
双Agent模式：同普通问答，显式调用双 Agent 流程

API 调用
GET /ask?question=你的问题&mode=normal
响应示例：
{
  "answer": "今天图书馆8点开门...",
  "source": "模型生成"
}

🔧 高级功能
双 Agent 协作（检索 + 口语化）
检索 Agent：从向量库召回相关知识点，整理为简洁要点。
生成 Agent：将要点转化为轻松亲切的学长学姐口吻回复，避免生硬书面语。
该设计在未进行微调时，也能获得接近微调风格的体验。

CoT 路线规划
当 mode=route 时，系统会强制 LLM 按 4 步思考：

拆解地点与时间
核对开放时间
排序并规划路线
补充注意事项

最终输出包含思考过程和最终回答，提升复杂行程问题的准确性。

Agent 工具
内置两个工具：
empty_classroom(building, time)：返回指定教学楼空教室（示例数据）
campus_weather()：返回当日天气（示例数据）

可轻松扩展更多工具。

LRU 缓存
相同问题（含模式）命中缓存后直接返回，响应头标记 source: "缓存命中"，缓存上限可在 config.py 调整。

🧪 LoRA 微调（可选）
项目支持基于 train_data.json 进行 LoRA 微调，以适应校园场景的口语风格。

数据格式
train_data.json 为 JSON 数组，每条包含 instruction 和 output 字段：

[
  {"instruction": "图书馆几点开门？", "output": "工作日8点开门..."},
  ...
]

训练
确保 config.py 中 MODEL_NAME 指定基础模型（如 deepseek-chat 或本地模型路径）。

根据需要调整 LoRA 超参（LORA_R, LORA_ALPHA 等）。

运行：
python train_lora.py
训练完成后，LoRA 权重保存至 ./lora_weights/。

使用微调模型
在 config.py 中设置：
USE_FINETUNE_MODEL = True
LORA_WEIGHT_PATH = "./lora_weights/"
重启服务，系统将自动加载微调后的模型进行推理。

⚙️ 配置详解
配置项	说明
API_KEY / BASE_URL / MODEL_NAME	在线 LLM 接口配置
EMBEDDING_MODEL_NAME	嵌入模型名称，默认 shibing624/text2vec-base-chinese
DB_PATH	向量库持久化目录
CACHE_MAX	LRU 缓存最大条目数
USE_FINETUNE_MODEL	是否启用本地微调模型
LORA_WEIGHT_PATH	LoRA 权重保存/加载路径
LORA_R / LORA_ALPHA / LORA_DROPOUT	LoRA 超参数
MAX_TRAIN_EPOCH / TRAIN_BATCH_SIZE	训练轮次和批次大小

📦 依赖说明
核心依赖列表（见 requirements.txt）：

LangChain 生态：langchain, langchain-openai, langchain-community, chromadb
FastAPI + Uvicorn：Web 服务
Sentence-Transformers：文本嵌入
PEFT / Transformers / TRL：LoRA 微调支持
Torch：深度学习后端

❗ 注意事项
向量库构建：若修改知识文本，需重新运行 build_db.py。
微调资源：本地微调需要一定显存（建议 ≥ 8GB），可根据硬件调整 TRAIN_BATCH_SIZE。
模型兼容性：使用在线 API 时需确保网络通畅；本地微调模型需与 MODEL_NAME 基础模型一致。
前端静态文件：templates/index.html 必须存在，否则根路由会报错。

🤝 贡献
欢迎提交 Issue 或 Pull Request，共同完善校园智能助手。

📄 许可证
本项目仅供学习交流使用，请勿用于商业用途。

📄 许可证
本项目仅供学习交流使用，请勿用于商业用途。

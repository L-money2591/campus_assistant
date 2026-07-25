import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from collections import OrderedDict
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config import *

# 微调本地模型依赖（仅USE_FINETUNE_MODEL=True时生效）
finetune_model = None
finetune_tokenizer = None
if USE_FINETUNE_MODEL:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="校园智能助手")
MAX_INPUT_LENGTH = 200

# ========== LLM入口（支持在线API / 本地微调LoRA模型） ==========
def get_llm(t=0.3):
    global finetune_model, finetune_tokenizer
    if USE_FINETUNE_MODEL:
        if finetune_model is None:
            base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
            finetune_model = PeftModel.from_pretrained(base_model, LORA_WEIGHT_PATH)
            finetune_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        def local_infer(prompt):
            inputs = finetune_tokenizer(prompt, return_tensors="pt")
            outputs = finetune_model.generate(**inputs, temperature=t, max_new_tokens=512)
            return finetune_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return local_infer
    else:
        return ChatOpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
            model=MODEL_NAME,
            temperature=t,
            request_timeout=30
        )

# ========== LRU缓存模块 ==========
cache = OrderedDict()
def get_cache(question):
    if question in cache:
        cache.move_to_end(question)
        return cache[question]
    return None
def set_cache(question, answer):
    if question in cache:
        cache.move_to_end(question)
    else:
        if len(cache) >= CACHE_MAX:
            cache.popitem(last=False)
    cache[question] = answer

# ========== RAG向量检索 ==========
try:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})
except Exception as e:
    print(f"向量库初始化失败：{e}")
    embeddings = None
    retriever = None
def rag_search(query):
    if not retriever:
        return "知识库暂不可用"
    try:
        docs = retriever.get_relevant_documents(query)
        return "\n".join([d.page_content for d in docs])
    except Exception as e:
        return f"检索出错：{str(e)}"

# ========== CoT思维链路线规划 ==========
def cot_route_plan(query):
    cot_prompt = f"""
    你是校园助手，处理行程问题必须按4步思考：
    1. 拆解用户要去的地点和时间
    2. 核对每个地点的开放时间
    3. 按顺路原则排先后顺序，给出路线
    4. 补充注意事项
    先写【思考过程】，再写【最终回答】。
    用户问题：{query}
    参考信息：{rag_search(query)}
    """
    llm = get_llm(0.2)
    if USE_FINETUNE_MODEL:
        return llm(cot_prompt)
    return llm.invoke(cot_prompt).content

# ========== Agent工具函数 ==========
@tool
def empty_classroom(building: str, time: str) -> str:
    """查询指定教学楼指定时间段的空教室"""
    return f"{building} {time} 空教室：102、205、308"
@tool
def campus_weather() -> str:
    """查询校园今日天气"""
    return "今日晴，26℃，微风，适合出门"
tools = [empty_classroom]
def get_agent():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是校园助手，优先用工具查信息，回答口语化"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(get_llm(0.1), tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)

# ========== 双Agent协作（检索+口语生成，模拟微调效果） ==========
def retrieval_agent(query):
    info = rag_search(query)
    prompt = f"把下面的信息整理成3条以内的要点，不要多余话：\n{info}"
    llm = get_llm(0)
    if USE_FINETUNE_MODEL:
        return llm(prompt)
    return llm.invoke(prompt).content

def generate_agent(query, points):
    prompt = f"""
    用户校园提问：{query}
    参考事实要点：{points}
    要求：
    1. 完全模仿在校学长学姐口语，轻松亲切，禁止官方生硬书面语；
    2. 简短自然，适当使用校园生活化语气词；
    3. 不编造信息，全部基于给你的要点；
    """
    llm = get_llm(0.7)
    if USE_FINETUNE_MODEL:
        return llm(prompt)
    return llm.invoke(prompt).content

def double_agent_answer(query):
    points = retrieval_agent(query)
    return generate_agent(query, points)

# ========== 后端接口 ==========
@app.get("/ask")
def ask(question: str, mode: str = "normal"):
    if not question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    if len(question) > MAX_INPUT_LENGTH:
        raise HTTPException(status_code=400, detail=f"问题长度不能超过{MAX_INPUT_LENGTH}字")
    cache_key = f"{question}_{mode}"
    cached = get_cache(cache_key)
    if cached:
        return {"answer": cached, "source": "缓存命中"}
    try:
        if mode == "route":
            ans = cot_route_plan(question)
        elif mode == "tool":
            agent = get_agent()
            ans = agent.invoke({"input": question})["output"]
        elif mode == "double" or mode == "normal":
            ans = double_agent_answer(question)
        else:
            ans = double_agent_answer(question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成回答失败：{str(e)}")
    set_cache(cache_key, ans)
    return {"answer": ans, "source": "模型生成"}

# 前端页面路由
@app.get("/", response_class=HTMLResponse)
async def chat_page():
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="templates文件夹下前端页面缺失")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
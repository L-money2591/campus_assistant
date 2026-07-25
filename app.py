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

app = FastAPI(title="校园智能助手")
MAX_INPUT_LENGTH = 200

# ========== LLM入口（支持在线API / 本地微调LoRA模型） ==========
def get_llm(t=0.3):
    """
    返回一个兼容的 LLM 对象：
    - 若 USE_FINETUNE_MODEL 为 True，返回 LocalLLM（包含 invoke() 与 __call__()，invoke 返回有 .content 属性的对象）
    - 否则返回 ChatOpenAI 实例（langchain_openai）
    """
    global finetune_model, finetune_tokenizer
    if USE_FINETUNE_MODEL:
        # 延迟加载微调模型
        if finetune_model is None:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
            base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
            finetune_model = PeftModel.from_pretrained(base_model, LORA_WEIGHT_PATH)
            finetune_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        def local_infer_str(prompt: str, temperature: float = t, max_new_tokens: int = 512) -> str:
            inputs = finetune_tokenizer(prompt, return_tensors="pt")
            outputs = finetune_model.generate(**inputs, temperature=temperature, max_new_tokens=max_new_tokens)
            return finetune_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 包装成具有 invoke()/__call__() 且 invoke 返回对象包含 .content 的类，兼容 app 中对 llm.invoke(...).content 的使用
        class LocalLLM:
            def __init__(self, infer_func):
                self._infer = infer_func

            def invoke(self, prompt: str):
                class R:
                    def __init__(self, content):
                        self.content = content
                return R(self._infer(prompt))

            def __call__(self, prompt: str):
                # 返回字符串，便于直接调用 llm(prompt)
                return self._infer(prompt)

        return LocalLLM(local_infer_str)
    else:
        # 远程/在线模型（langchain ChatOpenAI）
        return ChatOpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
            model=MODEL_NAME,
            temperature=t,
            request_timeout=30
        )

def call_llm(llm, prompt: str) -> str:
    """
    统一调用 llm，并返回字符串结果。
    兼容：llm.invoke(...).content, llm(...), llm.invoke(...)=str, llm(...) 返回 dict 等。
    """
    try:
        if hasattr(llm, "invoke"):
            out = llm.invoke(prompt)
        else:
            out = llm(prompt)
    except TypeError:
        # 有些 LLM 可能需要 named args or dict input; fallback:
        try:
            out = llm({"input": prompt})
        except Exception as e:
            raise

    # 解析常见返回格式
    if out is None:
        return ""
    if isinstance(out, str):
        return out
    if hasattr(out, "content"):
        return out.content
    if isinstance(out, dict):
        # 常见结构检查
        for key in ("output", "text", "answer", "content"):
            if key in out:
                val = out[key]
                if isinstance(val, str):
                    return val
        # 若字典中含 choices
        if "choices" in out and isinstance(out["choices"], list) and len(out["choices"]) > 0:
            c = out["choices"][0]
            if isinstance(c, dict) and "text" in c:
                return c["text"]
    # 回退到字符串化
    return str(out)

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
    return call_llm(llm, cot_prompt)

# ========== Agent工具函数 ==========
@tool
def empty_classroom(building: str, time: str) -> str:
    """查询指定教学楼指定时间段的空教室"""
    return f"{building} {time} 空教室：102、205、308"

@tool
def campus_weather() -> str:
    """查询校园今日天气"""
    return "今日晴，26℃，微风，适合出门"

# 将所有工具放入列表
tools = [empty_classroom, campus_weather]

def get_agent():
    """
    根据 langchain 版本差异，create_openai_tools_agent 可能存在差异。
    这里尽量传入一个 llm 对象（如果是本地微调，get_llm 已经返回兼容的包装对象）。
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是校园助手，优先用工具查信息，回答口语化"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    llm_obj = get_llm(0.1)
    try:
        agent = create_openai_tools_agent(llm_obj, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=False)
    except Exception:
        # 如果 create_openai_tools_agent/AgentExecutor 的签名不匹配，尝试返回 agent object 作为-is（有可能 create_openai_tools_agent 已返回 AgentExecutor）
        try:
            return create_openai_tools_agent(llm_obj, tools, prompt)
        except Exception as e:
            # 最后回退为 None（调用端会做处理）
            print(f"无法创建 agent：{e}")
            return None

def run_agent(agent, question: str) -> str:
    """
    防御性地运行 agent，兼容多种 agent API：
    - agent.run(question)
    - agent.invoke({'input': question})
    - agent({'input': question})
    - agent.execute / agent.__call__
    最终返回字符串。
    """
    if agent is None:
        return "Agent 不可用"
    # 尝试多种调用方式
    try:
        if hasattr(agent, "run"):
            out = agent.run(question)
            if isinstance(out, str):
                return out
        if hasattr(agent, "invoke"):
            out = agent.invoke({"input": question})
            # out 可能是 dict 或对象
            if isinstance(out, dict) and "output" in out:
                return out["output"]
            if hasattr(out, "content"):
                return out.content
            if isinstance(out, str):
                return out
        # 直接调用
        try:
            out = agent({"input": question})
            if isinstance(out, dict):
                for k in ("output", "text", "answer"):
                    if k in out and isinstance(out[k], str):
                        return out[k]
            if isinstance(out, str):
                return out
        except Exception:
            pass
    except Exception as e:
        return f"Agent 调用出错：{e}"
    # 最后回退
    return "Agent 未返回结果"

# ========== 双Agent协作（检索+口语生成，模拟微调效果） ==========
def retrieval_agent(query):
    info = rag_search(query)
    prompt = f"把下面的信息整理成3条以内的要点，不要多余话：\n{info}"
    llm = get_llm(0)
    return call_llm(llm, prompt)

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
    return call_llm(llm, prompt)

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
            ans = run_agent(agent, question)
        elif mode == "double" or mode == "normal":
            ans = double_agent_answer(question)
        else:
            ans = double_agent_answer(question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成回答失败：{str(e)}")
    set_cache(cache_key, ans)
    return {"answer": ans, "source": "模型生成"}

# 前端页面路由（直接读取根目录 index.html）
@app.get("/", response_class=HTMLResponse)
async def chat_page():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="前端页面 index.html 缺失（请确保仓库根目录存在 index.html 或修改路径）")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

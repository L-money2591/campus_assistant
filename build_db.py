import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import DB_PATH, EMBEDDING_MODEL_NAME

loaders = [
    TextLoader("./图书馆.txt", encoding="utf-8"),
    TextLoader("./食堂快递.txt", encoding="utf-8")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
# 修复：传入docs参数
split_docs = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
Chroma.from_documents(split_docs, embeddings, persist_directory=DB_PATH)
print("✅ 向量库构建完成！")
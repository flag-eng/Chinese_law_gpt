# build_db.py
from modules.rag_engine import RAGEngine

if __name__ == "__main__":
    rag = RAGEngine()
    rag.build_db_from_txt("data")
    print("向量数据库构建完成！")

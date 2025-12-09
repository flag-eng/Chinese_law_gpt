# app.py
import json
from typing import Optional
import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from modules.rag_engine import RAGEngine
import uuid
app = FastAPI(title="法律问答系统 API", version="1.0")

# 1. 获取项目根目录路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. 挂载静态文件目录
static_dir = os.path.join(BASE_DIR, "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

rag = RAGEngine()  # 启动时加载向量数据库

# [修改] 请求体增加 session_id 字段
class QARequest(BaseModel):
    question: str
    # [修改] 使用 Optional[str] 明确允许传入 null
    session_id: Optional[str] = None

@app.post("/ask")
def ask_law(req: QARequest):
    # 1. 如果前端没传 session_id，生成一个新的
    current_session_id = req.session_id
    if not current_session_id:
        current_session_id = str(uuid.uuid4())
        print(f"New session started: {current_session_id}")

    def generate_stream():
        try:
            # 2. [关键] 在 start 消息中把 session_id 返还给前端
            # 前端收到后，必须把它存起来，下次发请求时带上！
            start_payload = {
                'type': 'start',
                'question': req.question,
                'session_id': current_session_id
            }
            yield f"data: {json.dumps(start_payload, ensure_ascii=False)}\n\n"

            # 3. [关键] 调用 rag.answer 时传入 session_id
            for response in rag.answer(req.question, session_id=current_session_id):
                yield f"data: {json.dumps(response, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

        except Exception as e:
            error_msg = f"处理问题时发生错误: {str(e)}"
            print(f"Error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# 3. 访问根路径时直接返回前端页面
@app.get("/")
def root():
    # 优先尝试返回 static/index.html
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"msg": "Legal QA API is running. Please place index.html in 'static' folder."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
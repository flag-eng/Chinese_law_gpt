# modules/hw_api.py
import requests
import json
import yaml
import os

# 加载配置文件
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

HW_API_KEY = config["HW_API_KEY"]
DEEPSEEK_URL = config["DEEPSEEK_URL"]
RERANK_URL = config["RERANK_URL"]
DEEPSEEK_MODEL = config["DEEPSEEK_MODEL"]
RERANK_MODEL = config["RERANK_MODEL"]

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {HW_API_KEY}"
}


############################
# DeepSeek V3 文本生成（流式）
############################
def generate_api_stream(messages: list):
    """
    返回一个生成器（generator）
    每次 yield 一小段模型返回的内容
    messages: [{"role": "system",...}, {"role": "user",...}, {"role": "assistant",...}]
    """

    payload = {
        "model": DEEPSEEK_MODEL,
        "stream": True,
        "messages": messages, # [修改] 直接使用传入的完整消息列表
        "thinking": {"type": "enabled"}
    }

    try:
        # 开启流式请求
        with requests.post(
            DEEPSEEK_URL,
            headers=HEADERS,
            data=json.dumps(payload),
            verify=False,
            timeout=120,  # 流式生成需要更长 timeout
            stream=True   # <<< 核心：开启长连接流式读取
        ) as resp:

            resp.raise_for_status()

            # 按行读取 SSE（Server-Sent Events）
            for line in resp.iter_lines(decode_unicode=True):

                if not line:
                    continue

                # 流式内容以 data: 开头
                if line.startswith("data: "):
                    data = line[len("data: "):].strip()

                    # 结束标记
                    if data == "[DONE]":
                        break

                    try:
                        j = json.loads(data)
                        # 提取内容（不同模型格式不同，这里用 choices 兼容）
                        if "choices" in j and len(j["choices"]) > 0:
                            delta = j["choices"][0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield content
                    except Exception:
                        # JSON 解码失败，不影响继续读
                        continue

    except requests.exceptions.RequestException as e:
        yield f"[流式请求异常] {str(e)}"


############################
# [新增] 普通文本生成（非流式，用于 Query Rewrite）
############################
def generate_api_sync(messages: list):
    """
    同步调用接口，直接返回完整的字符串内容
    用于重写问题等中间步骤
    """
    payload = {
        "model": DEEPSEEK_MODEL,
        "stream": False,  # 关闭流式
        "messages": messages,
        "temperature": 0.1  # 重写任务需要低随机性，越精确越好
    }

    try:
        resp = requests.post(
            DEEPSEEK_URL,
            headers=HEADERS,
            data=json.dumps(payload),
            verify=False,
            timeout=30
        )
        resp.raise_for_status()
        j = resp.json()

        # 解析返回结果
        if "choices" in j and len(j["choices"]) > 0:
            return j["choices"][0]["message"]["content"].strip()

        return ""

    except Exception as e:
        print(f"[API Error] Rewrite failed: {e}")
        return ""  # 出错时返回空字符串，后续逻辑会回退到原始问题


############################
# BGE 重排
############################
def rerank_api(query: str, docs: list):
    if not docs:
        return []

    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": docs
    }

    try:
        resp = requests.post(
            RERANK_URL,
            headers=HEADERS,
            data=json.dumps(payload),
            verify=False,
            timeout=30
        )
        resp.raise_for_status()
        j = resp.json()

        # 官方格式支持
        if "results" in j:
            return [item["relevance_score"] for item in j["results"]]

        return [0.0] * len(docs)
    except requests.exceptions.RequestException as e:
        # 如果重排失败，返回默认分数
        return [0.0] * len(docs)

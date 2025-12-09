# modules/rag_engine.py
import yaml
import os
import sys
from collections import defaultdict
# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from concurrent.futures import ThreadPoolExecutor, as_completed
from modules.hw_api import generate_api_stream, rerank_api, generate_api_sync
from modules.embedder import Embedder
from modules.vector_db import VectorDB
import numpy as np

# 加载配置文件
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


class RAGEngine:
    def __init__(self, model_name="shibing624/text2vec-base-chinese"):
        self.model_name = model_name
        self.embedder = Embedder(model_name)
        self.dim = self.embedder.encode(["test"]).shape[1]
        self.vector_db = VectorDB(dim=self.dim)  # 若已有 index，则自动加载

        # [新增] 内存中的对话历史存储
        # 结构: {"session_id_1": [{"role": "user", "content": "..."}, ...], ...}
        self.chat_history = defaultdict(list)
        # 历史记录最大保留轮数（避免 Token 溢出）
        self.max_history_len = 10

    # ------------------------- #
    # 构建向量数据库（build_db.py 调用）
    # ------------------------- #
    def build_db_from_txt(self, folder="data", max_workers=8, batch_size=32, min_line_length=5):
        """
        使用多线程逐行读取txt文件并构建向量数据库

        Args:
            folder: 数据文件夹路径
            max_workers: 最大线程数，默认8
            batch_size: embedding批量处理大小，默认32
            min_line_length: 最小行长度（字符数），过滤过短的行，默认5。设为0则不过滤
        """
        if not os.path.exists(folder):
            raise FileNotFoundError(f"数据文件夹不存在: {folder}")

        # 收集所有txt文件路径
        txt_files = []
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                path = os.path.join(folder, filename)
                txt_files.append(path)

        if not txt_files:
            raise ValueError(f"在 {folder} 文件夹中未找到有效的 .txt 文件")

        print(f"找到 {len(txt_files)} 个文本文件，开始多线程逐行读取...")

        # 多线程逐行读取文件
        def read_file_lines(filepath):
            """逐行读取文件，返回行列表"""
            try:
                lines = []
                filename = os.path.basename(filepath)
                with open(filepath, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, start=1):
                        line = line.strip()
                        # 过滤空行，如果设置了最小长度则过滤过短的行
                        if line and (min_line_length == 0 or len(line) >= min_line_length):
                            lines.append({
                                "text": line,
                                "file": filename,
                                "line": line_num
                            })
                return lines
            except Exception as e:
                print(f"读取文件失败 {filepath}: {str(e)}")
                return []

        # 使用多线程读取所有文件的行
        all_texts = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有读取任务
            future_to_file = {executor.submit(read_file_lines, filepath): filepath
                              for filepath in txt_files}

            # 收集结果
            completed = 0
            for future in as_completed(future_to_file):
                lines = future.result()
                all_texts.extend(lines)
                completed += 1
                if completed % 10 == 0:
                    print(f"已读取 {completed}/{len(txt_files)} 个文件，累计 {len(all_texts)} 行...")

        if not all_texts:
            raise ValueError(f"在 {folder} 文件夹中未找到有效的文本内容")

        print(f"文件读取完成，共 {len(all_texts)} 行有效内容，开始生成向量...")

        # 批量生成embeddings（sentence_transformers支持批量处理，效率更高）
        all_embeddings = []
        text_list = [t["text"] for t in all_texts]

        # 分批处理，避免内存溢出
        total_batches = (len(text_list) + batch_size - 1) // batch_size
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i + batch_size]
            batch_embeddings = self.embedder.encode(batch)
            all_embeddings.append(batch_embeddings)
            batch_num = i // batch_size + 1
            if batch_num % 10 == 0 or batch_num == total_batches:
                print(f"已处理 {min(i + batch_size, len(text_list))}/{len(text_list)} 行文本的向量化...")

        # 合并所有embeddings
        embeddings = np.vstack(all_embeddings)

        print("开始添加到向量数据库...")
        self.vector_db.add(embeddings, all_texts)
        self.vector_db.save()
        print(f"数据库构建完成！共加入 {len(all_texts)} 行法律内容（来自 {len(txt_files)} 个文件）")

    # ------------------------- #
    # [修改] 独立的问题重写方法
    # ------------------------- #
    def _rewrite_question(self, question, history):
        """
        利用大模型，结合历史记录重写当前问题
        """
        if not history:
            print(f"\n{'=' * 20} [1. 问题处理] {'=' * 20}")
            print(f"无需重写 (无历史记录)，直接使用原始问题: {question}")
            return question

        short_history = history[-4:]
        messages = [{"role": "system", "content": config["REWRITE_PROMPT"]}]
        messages.extend(short_history)
        messages.append({"role": "user", "content": question})

        print(f"\n{'=' * 20} [1. 问题重写] {'=' * 20}")
        print("正在调用模型进行重写...")
        rewritten = generate_api_sync(messages)

        if not rewritten:
            print("重写失败或返回为空，使用原问题。")
            return question

        print(f"原始问题: {question}")
        print(f"重写结果: {rewritten}")
        return rewritten

    # ------------------------- #
    # [修改] 主问答逻辑
    # ------------------------- #
    def answer(self, question, session_id="default"):
        if not question or not question.strip():
            yield {"type": "error", "content": "问题不能为空", "sources": []}
            return

        # 获取历史记录
        history = self.chat_history[session_id]

        # === [核心步骤 1] Query Rewriting (问题重写) ===
        search_query = self._rewrite_question(question, history)

        # 打印当前使用的检索词
        print(f"最终用于检索的 Query: [{search_query}]")

        q_emb = self.embedder.encode([search_query])

        # === [核心步骤 2] 向量检索 ===
        print(f"\n{'=' * 20} [2. 向量检索] {'=' * 20}")
        candidates = self.vector_db.search(q_emb, top_k=5)
        print(f"向量库初筛回 {len(candidates)} 条记录:")
        for idx, cand in enumerate(candidates):
            print(f"  [候选 {idx + 1}] {cand['file']} (第{cand['line']}行): {cand['text'][:30]}...")

        if not candidates:
            print("未找到相关内容，流程终止。")
            yield {"type": "error", "content": "未找到相关内容", "sources": []}
            return

        # === [核心步骤 3] 重排 ===
        print(f"\n{'=' * 20} [3. Rerank 重排] {'=' * 20}")
        scores = rerank_api(search_query, [c["text"] for c in candidates])

        # 将候选与分数打包
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

        # 打印重排后的分数情况
        for idx, (cand, score) in enumerate(ranked):
            print(f"  [排名 {idx + 1}] Score: {score:.4f} | {cand['text'][:20]}...")

        top_sources = [item[0] for item in ranked[:3]]
        context = "\n".join([item["text"] for item in top_sources])
        print(f"最终选取 Top {len(top_sources)} 作为上下文输入给大模型。")

        # 发送来源给前端
        for item in top_sources:
            yield {
                "type": "source",
                "title": f"{item['file']} (第{item['line']}行)",
                "content": item['text']
            }

        # === [核心步骤 4] 生成回复 ===
        print(f"\n{'=' * 20} [4. 大模型生成] {'=' * 20}")
        current_user_content = config["USER_PROMPT_TEMPLATE"].format(
            context=context,
            question=question
        )

        messages = [{"role": "system", "content": config["SYSTEM_PROMPT"]}]
        messages.extend(history)
        messages.append({"role": "user", "content": current_user_content})

        full_response_content = ""
        print("开始流式输出回答...")
        try:
            for chunk in generate_api_stream(messages):
                full_response_content += chunk
                # 这里可以选择是否在控制台实时打印 chunk，为了日志整洁，通常不打印碎片
                # print(chunk, end="", flush=True)
                yield {"type": "chunk", "content": chunk}

            # 生成结束后打印完整回复
            print(f"\n[生成结束] 完整回复内容:\n{full_response_content}")
            print(f"{'=' * 20} [流程结束] {'=' * 20}\n")

            # 更新记忆
            self.chat_history[session_id].append({"role": "user", "content": question})
            self.chat_history[session_id].append({"role": "assistant", "content": full_response_content})

            if len(self.chat_history[session_id]) > self.max_history_len * 2:
                self.chat_history[session_id] = self.chat_history[session_id][-(self.max_history_len * 2):]

        except Exception as e:
            print(f"[Error] 生成过程中发生异常: {e}")
            yield {"type": "error", "content": f"生成失败: {str(e)}"}
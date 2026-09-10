from openai import OpenAI

from .config import API_KEY, BASE_URL, MODEL_NAME


class TextFormatter:
    def __init__(self):
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL
        )

    def format_text(self, text: str):
        if not text:
            return ""

        prompt = f"""
你是一名专业会议记录整理助手。

请优化下面的语音识别文本：

要求：
1. 根据语义添加中文标点（，。？！）。
2. 根据停顿和语义自动分段换行。
3. 修正明显的口语转写错误。
4. 不增加不存在的信息。
5. 保持原始含义。

原始文本：
{text}

请只输出整理后的文本。
"""

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )

        return response.choices[0].message.content

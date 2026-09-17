from openai import OpenAI
from .config import API_KEY, BASE_URL, MODEL_NAME
from .prompt import SUMMARY_PROMPT


class AISummary:
    def __init__(self):
        # 未配置 API Key 时置为 None，generate 返回提示信息，不发起网络请求
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL) if API_KEY else None

    def generate(self, text):
        if not text:
            return "暂无文本内容"
        if self.client is None:
            return "未配置 OPENAI_API_KEY，无法生成 AI 摘要。"

        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个智能会议助手"
                    },
                    {
                        "role": "user",
                        "content": SUMMARY_PROMPT.format(text=text)
                    }
                ],
                temperature=0.3
            )
            return response.choices[0].message.content or "（模型未返回内容）"
        except Exception as e:
            return f"AI 摘要生成失败：{e}"

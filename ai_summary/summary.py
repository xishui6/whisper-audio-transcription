from openai import OpenAI
from .config import API_KEY, BASE_URL, MODEL_NAME
from .prompt import SUMMARY_PROMPT


class AISummary:
    def __init__(self):
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL
        )

    def generate(self, text):
        if not text:
            return "暂无文本内容"

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

        return response.choices[0].message.content

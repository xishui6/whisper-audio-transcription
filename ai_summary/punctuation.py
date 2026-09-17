from openai import OpenAI

from .config import API_KEY, BASE_URL, MODEL_NAME


class TextFormatter:
    """在线 LLM 文本整理（可选）：配置了 OPENAI_API_KEY 时用于精修标点与分段。"""

    def __init__(self):
        # 未配置 API Key 时置为 None，format_text 原样返回文本，不发起网络请求
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL) if API_KEY else None

    def format_text(self, text: str):
        if not text:
            return ""
        if self.client is None:
            return text

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

        try:
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
            return response.choices[0].message.content or text
        except Exception:
            # 网络 / 鉴权 / 服务异常时回退到原始文本，不阻断转写主流程
            return text


class RulePunctuator:
    """离线规则标点恢复：基于词级时间戳与中文口语语气词 / 连接词规则的断句器。

    输入：words = [(start, end, text), ...]（秒级时间戳，来自 faster-whisper word_timestamps）
    输出：句子列表，每句为 dict：
        {"text": str, "punct": str, "start": float, "end": float}
    """

    # ---- 断句信号 ----
    # 句尾语气词：词以这些字结尾即结束当前句（了 是完成态标记，也常是意群结尾）
    _TAIL_PARTICLE = ("吗", "呢", "么", "嘛", "吧", "啊", "呀", "啦", "哦", "嘞",
                      "哈", "哇", "唉", "哟", "喽", "了")
    # 独立成词且当前句为空时，附加到上一句（whisper 常把语气词拆成单独 token）
    _ATTACH_TAIL = ("吗", "呢", "么", "嘛", "吧", "啊", "呀", "啦", "哦", "嘞",
                    "哈", "哇", "唉", "哟", "喽")
    _Q_PARTICLE = ("吗", "呢", "么", "嘛")
    # 疑问词：句内包含即判为疑问句
    _QUESTION_WORDS = ("什么", "怎么", "为啥", "干嘛", "为什么", "哪", "谁", "几", "多少", "啥",
                       "是不是", "有没有", "能不能", "可不可以", "没有")
    # 句尾感叹词
    _EXCLAIM_TAIL = ("啊", "呀", "啦", "哇", "唉", "哟", "嘿", "哈", "喽")
    # 口语连接词：作为词开头出现时，在它之前断句并并入前句（前句句号改逗号）
    _CONNECTORS = ("然后", "但是", "不过", "所以", "因为", "如果", "而且", "其实", "可是",
                   "后来", "结果", "就是", "接着", "于是", "再说", "那么", "总之", "反正",
                   "毕竟", "再然后", "但", "那")
    # 以这些词开头的词不视为连接词（代词/指示词误报）
    _CONNECTOR_EXCLUDE = ("那个", "那些", "那里", "那边", "那家", "那时", "那会", "那阵", "但是呢")
    # 句首填充词：单列成短句，后面接逗号
    _FILLER = ("喂", "嗯", "哎", "嗨", "哦", "哟", "好", "行", "对", "啊")

    GAP_PERIOD = 0.60     # 停顿 >= 此值 → 断句
    MAX_SENT_CHARS = 40   # 单句最大字符数，超限强制在词边界拆分
    MAX_MERGE_CHARS = 60  # 连接词合并后的最大句长
    MIN_CONNECTOR_SENT = 2  # 连接词/填充词断句要求前句最少字数

    # ---------- 断句 ----------

    def split_sentences(self, words):
        sentences = []
        cur = []
        cur_chars = 0
        prev_end = None

        def close():
            nonlocal cur, cur_chars
            if cur:
                sentences.append(cur)
                cur = []
                cur_chars = 0

        for raw in words:
            start, end, text = raw
            text = text.strip()
            if not text:
                continue
            gap = (start - prev_end) if prev_end is not None else 0.0
            prev_end = end
            w = (start, end, text)

            # G. 独立语气词/“没有” 且当前句为空 → 附加到上一句（避免“…是吧”被拆开）
            if not cur and sentences and (text in self._ATTACH_TAIL or text in ("没有", "没点")):
                sentences[-1].append(w)
                continue

            # D. 句首填充词（当前句为空）→ 独立短句（标点“，”）
            if not cur and text in self._FILLER:
                cur = [w]
                cur_chars = len(text)
                close()
                continue

            # B/C. 连接词或填充词开头的词，且当前句已有足够内容 → 在词前断句，该词开新句
            if cur and (self._is_connector_word(text) or text in self._FILLER) and (
                    cur_chars >= self.MIN_CONNECTOR_SENT
                    or (len(cur) == 1 and cur[0][2] in self._FILLER)):
                close()
                cur = [w]
                cur_chars = len(text)
                continue

            cur.append(w)
            cur_chars += len(text)

            # A. 当前词以句尾语气词结尾 / 是“没有”“没点” → 立即断句
            if text[-1] in self._TAIL_PARTICLE or text in ("没有", "没点"):
                close()
                continue

            # E. 停顿时长
            if gap >= self.GAP_PERIOD and len(cur) >= 2:
                close()
                continue

            # F. 句长上限
            if cur_chars >= self.MAX_SENT_CHARS:
                close()

        close()
        return sentences

    @classmethod
    def _is_connector_word(cls, text):
        if any(text.startswith(e) for e in cls._CONNECTOR_EXCLUDE):
            return False
        return any(text.startswith(c) for c in cls._CONNECTORS)

    # ---------- 标点 ----------

    def _choose_punct(self, sent_words, whole_text):
        # 疑问句：含疑问语气词或疑问词
        if any(q in whole_text for q in self._Q_PARTICLE) or any(
                q in whole_text for q in self._QUESTION_WORDS):
            return "？"
        # 感叹句：句尾为感叹词
        tail = whole_text[-1]
        if tail in self._EXCLAIM_TAIL:
            return "！"
        # 填充词短句 → 逗号
        if len(sent_words) == 1 and whole_text in self._FILLER:
            return "，"
        return "。"

    # ---------- 主流程 ----------

    def punctuate(self, words):
        sents = self.split_sentences(words)
        out = []
        for raw in sents:
            start = raw[0][0]
            end = raw[-1][1]
            text = "".join(w[2] for w in raw)
            punct = self._choose_punct(raw, text)
            out.append({"text": text, "punct": punct, "start": start, "end": end})

        # 连接词合并：某句以连接词开头且前句标点是句号 → 并入前句（前句句号改逗号）
        merged = []
        for s in out:
            if merged and self._is_connector_word(s["text"]):
                prev = merged[-1]
                if prev["punct"] in ("。", "，") and \
                        len(prev["text"]) + len(s["text"]) <= self.MAX_MERGE_CHARS:
                    prev["text"] += "，" + s["text"]
                    prev["punct"] = s["punct"]
                    prev["end"] = s["end"]
                    continue
            merged.append(dict(s))
        return merged

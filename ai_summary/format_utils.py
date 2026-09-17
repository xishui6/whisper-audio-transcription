# -*- coding: utf-8 -*-
"""说话人分段与最终文本组装。

核心思路：
1. faster-whisper 转写时开启 word_timestamps，得到每个词的 (start, end, text)。
2. 本地说话人分离（diarization）给出 [(start, end, speaker_id), ...]。
3. 每个词按时间戳归属到某个说话人 → 句子带说话人标签。
4. 说话人切换处换行（行首加「说话人 N」标签）；同一说话人的连续话语合并成一行。
5. 若说话人分离不可用（模型未装 / 单说话人），退化为按停顿分段。
"""

import re

from .punctuation import RulePunctuator

# 说话人标签模板
SPEAKER_LABEL = "【说话人 {}】"
# 无说话人信息时的段落分界停顿（秒）
PARAGRAPH_GAP = 1.2
# 无说话人信息时一段最多包含的句子数
MAX_SENT_PER_PARA = 3

_REFINE_PROMPT = """你是语音转写文本整理助手。下面是语音转写文本，已经按说话人分段：每行是一个说话人的话，行首有【说话人 N】标签。

要求：
1. 保留所有【说话人 N】标签、每一行的顺序和换行结构，不要合并或拆分行。
2. 只对每行内部优化：修正中文标点（，。？！），修正明显的口语转写错误。
3. 不增删信息，不改变说话人归属。
直接输出整理后的完整文本，不要添加任何解释。"""


def refine_with_llm(text, client, model="gpt-4o-mini"):
    """用 LLM 精修标点与口语（保持行结构与说话人标签）。

    校验：行数一致且每行行首的【说话人 N】标签序列一致才接受结果，否则回退原文。
    """
    if not text or client is None:
        return text
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _REFINE_PROMPT + "\n\n" + text}],
            temperature=0.2,
        )
        out = (resp.choices[0].message.content or "").strip()
    except Exception:
        return text
    if not out:
        return text
    # 结构校验：行数一致 + 行首标签序列一致（行内文本允许被修改，例如加标点）
    def _label(line):
        m = re.match(r"^\s*【说话人\s*\d+】", line)
        return m.group(0) if m else ""

    if (len(out.splitlines()) == len(text.splitlines())
            and [_label(l) for l in out.splitlines()]
            == [_label(l) for l in text.splitlines()]):
        return out
    return text


def _assign_speaker(word_start, word_end, diar_segments):
    """计算词时间区间与各说话人片段的重叠时长，返回重叠最大的说话人 id（或 None）。"""
    if not diar_segments:
        return None
    best_spk, best_overlap = None, 0.0
    for seg_start, seg_end, spk in diar_segments:
        ov = min(word_end, seg_end) - max(word_start, seg_start)
        if ov > best_overlap:
            best_overlap = ov
            best_spk = spk
    return best_spk


def _sentence_speaker(sentence, diar_segments):
    """句子级说话人：取句子内所有词的重叠说话人中占比最高的。"""
    if not diar_segments:
        return None
    votes = {}
    total = 0.0
    for w in sentence:
        spk = _assign_speaker(w[0], w[1], diar_segments)
        votes[spk] = votes.get(spk, 0.0) + (w[1] - w[0])
        total += (w[1] - w[0])
    if not votes:
        return None
    best = max(votes, key=votes.get)
    if votes.get(best, 0.0) / max(total, 1e-6) < 0.4:
        return None
    return best


def build_transcript(words, diar_segments, punctuator=None, llm_client=None,
                     llm_model="gpt-4o-mini"):
    """生成最终转写文本。

    words: [(start, end, text), ...]
    diar_segments: [(start, end, speaker_id), ...] 或 None
    llm_client: 配置了 API key 时的 OpenAI 客户端（用于标点精修，可选）
    返回: (text, has_speaker)
    """
    punctuator = punctuator or RulePunctuator()
    sentences = punctuator.punctuate(words)

    # 为每个句子打说话人标签
    sent_spk = []
    for s in sentences:
        spk = None
        if diar_segments:
            # 用句子时间窗内的词投票
            seg_words = [
                (max(s["start"], w[0]), min(s["end"], w[1]), w[2])
                for w in words if w[1] > s["start"] and w[0] < s["end"]
            ]
            if seg_words:
                spk = _sentence_speaker(seg_words, diar_segments)
        s["speaker"] = spk
        sent_spk.append(s)

    has_speaker = any(s["speaker"] is not None for s in sent_spk)

    if has_speaker:
        text = _assemble_with_speakers(sent_spk)
    else:
        text = _assemble_plain(sentences)

    # 可选的 LLM 标点精修（失败自动回退）
    if llm_client is not None and text:
        text = refine_with_llm(text, llm_client, llm_model)
    return text, has_speaker


def _assemble_with_speakers(sentences):
    """按说话人换行：说话人变化 → 换行；连续同说话人合并。"""
    lines = []
    cur_spk = None
    cur_text = ""
    for s in sentences:
        seg = s["text"] + s["punct"]
        spk = s.get("speaker")
        if spk is not None and spk != cur_spk:
            if cur_text:
                lines.append((cur_spk, cur_text))
            cur_text = seg
            cur_spk = spk
        else:
            cur_text += seg
    if cur_text:
        lines.append((cur_spk, cur_text))

    parts = []
    for spk, line in lines:
        if spk is not None:
            parts.append(SPEAKER_LABEL.format(spk) + line)
        else:
            parts.append(line)
    return "\n".join(parts)


def _assemble_plain(sentences):
    """无说话人信息：按停顿/句数分段落，段内句子连写。"""
    paras = []
    cur_para = []
    prev_end = None
    for s in sentences:
        gap = (s["start"] - prev_end) if prev_end is not None else 0.0
        if cur_para and (gap >= PARAGRAPH_GAP or len(cur_para) >= MAX_SENT_PER_PARA):
            paras.append(cur_para)
            cur_para = []
        cur_para.append(s)
        prev_end = s["end"]
    if cur_para:
        paras.append(cur_para)

    return "\n\n".join("".join(s["text"] + s["punct"] for s in para) for para in paras)

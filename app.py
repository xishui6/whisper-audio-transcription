import streamlit as st
import json
import os
import time
from datetime import datetime

from faster_whisper import WhisperModel
from openai import OpenAI

from ai_summary.config import API_KEY, BASE_URL, MODEL_NAME
from ai_summary.summary import AISummary
from ai_summary.punctuation import RulePunctuator
from ai_summary.diarize import SpeakerDiarizer
from ai_summary.format_utils import build_transcript

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "faster-whisper-large-v3")
AUDIO_FOLDER = os.path.join(BASE_DIR, "audios")
RECORD_FILE = os.path.join(BASE_DIR, "records.json")

st.set_page_config(page_title="Whisper AI Assistant", page_icon="🎧", layout="wide")

# 存放最近一次转写结果，供 rerun 后展示（保证历史记录/计数同步刷新）
if "last_result" not in st.session_state:
    st.session_state.last_result = None


@st.cache_resource
def load_model():
    # 优先 GPU（float16）；CUDA 不可用时自动回退 CPU（int8）
    try:
        return WhisperModel(MODEL_DIR, device="cuda", compute_type="float16")
    except Exception:
        return WhisperModel(MODEL_DIR, device="cpu", compute_type="int8")


@st.cache_resource
def load_summary():
    return AISummary()


@st.cache_resource
def load_diarizer():
    return SpeakerDiarizer()


model = load_model()
summary_model = load_summary()
diarizer = load_diarizer()
llm_client = None
llm_status = "未配置 OPENAI_API_KEY，标点使用本地规则引擎"
if API_KEY:
    # 短超时 + 不重试：网络不通时快速回退本地规则，不让用户久等
    try:
        llm_client = OpenAI(api_key=API_KEY, base_url=BASE_URL,
                            timeout=8, max_retries=0)
        llm_status = "已配置 AI 标点精修（网络不通时自动回退本地规则）"
    except Exception as e:
        llm_status = f"AI 标点精修初始化失败：{e}"

os.makedirs(AUDIO_FOLDER, exist_ok=True)
if not os.path.exists(RECORD_FILE):
    with open(RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False)

with open(RECORD_FILE, "r", encoding="utf-8") as f:
    history = json.load(f)

st.title("🎧 Whisper AI Assistant")
st.caption("语音转写 · 标点断句 · 说话人分段 · AI摘要")

# ---- 侧边栏：功能开关与状态 ----
st.sidebar.header("⚙️ 设置")
use_diar = st.sidebar.checkbox("说话人分离（按人换行）", value=True,
                               help="需要本地 CAM++ 模型，首次运行自动下载；CPU 上较慢")
st.sidebar.info(llm_status)

c1, c2, c3 = st.columns(3)
c1.metric("📁历史记录", len(history))
c2.metric("🤖AI能力", "标点+说话人")
c3.metric("🔒模式", "本地ASR")

upload = st.file_uploader(
    "上传音频",
    type=["mp3", "wav", "m4a", "aac", "flac", "ogg", "opus",
          "webm", "mp4", "amr", "wma", "aiff"],
)

if upload:
    path = os.path.join(AUDIO_FOLDER, os.path.basename(upload.name))
    with open(path, "wb") as f:
        f.write(upload.getbuffer())
    st.audio(path)

    if st.button("🚀开始转写"):
        # ---- 1. 转写（带词级时间戳） ----
        start = time.time()
        with st.spinner("正在转写..."):
            segments, info = model.transcribe(path, language="zh", beam_size=8,
                                              word_timestamps=True,
                                              vad_filter=True,
                                              initial_prompt="以下是普通话的句子。")
            segment_list = list(segments)
            raw_text = "".join(s.text for s in segment_list)
            words = []
            for seg in segment_list:
                for w in (seg.words or []):
                    words.append((w.start, w.end, w.word))
        asr_cost = round(time.time() - start, 2)

        # ---- 2. 说话人分离（可开关，失败自动跳过） ----
        diar_segments = None
        diar_cost = 0.0
        if use_diar:
            with st.spinner("正在识别说话人（首次需加载模型，较慢）..."):
                t0 = time.time()
                diar_segments = diarizer.diarize(path)
                diar_cost = round(time.time() - t0, 1)
                if diar_segments is None and diarizer.error:
                    st.warning(f"说话人分离不可用（{diarizer.error}），将按停顿分段。")

        # ---- 3. 组装：规则标点 + 说话人分段 + LLM 精修 ----
        with st.spinner("正在整理文本..."):
            formatted_text, has_speaker = build_transcript(
                words, diar_segments, RulePunctuator(),
                llm_client=llm_client, llm_model=MODEL_NAME,
            )

        record = {
            "filename": upload.name,
            "raw_text": raw_text,
            "formatted_text": formatted_text,
            "cost_time": asr_cost,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        history.append(record)
        with open(RECORD_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        msg = f"完成，转写耗时 {asr_cost}s"
        if has_speaker:
            msg += f"，说话人识别 {diar_cost}s"
        st.session_state.last_result = {
            "formatted_text": formatted_text,
            "raw_text": raw_text,
            "msg": msg,
        }
        st.rerun()

# ---- 最近一次转写结果展示 ----
res = st.session_state.last_result
if res is not None:
    tab1, tab2, tab3 = st.tabs(["📝整理文本", "🎤原始转写", "🤖AI摘要"])

    with tab1:
        st.text_area("", res["formatted_text"], height=280,
                     key="formatted_text_area")
        st.download_button("下载整理文本", res["formatted_text"],
                           file_name="formatted.txt")

    with tab2:
        st.text_area("", res["raw_text"], height=280, key="raw_text_area")

    with tab3:
        if st.button("生成摘要"):
            try:
                st.write(summary_model.generate(res["formatted_text"]))
            except Exception as e:
                st.error(f"AI 摘要生成失败：{e}")

    st.success(res["msg"])

st.divider()
st.subheader("📜历史记录")
n = len(history)
for i in range(n):
    item = history[n - 1 - i]  # 最新记录在前
    with st.expander(item["filename"]):
        st.write(item.get("formatted_text", item.get("trans_text", "")))
        if st.button("🗑️ 删除该记录", key=f"del_{n - 1 - i}",
                     help="点击删除这条记录"):
            del history[n - 1 - i]
            with open(RECORD_FILE, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            st.rerun()

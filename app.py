import streamlit as st
import json
import os
import time
from datetime import datetime
from faster_whisper import WhisperModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "faster-whisper-small")
AUDIO_FOLDER = os.path.join(BASE_DIR, "audios")
RECORD_FILE = os.path.join(BASE_DIR, "records.json")

st.set_page_config(
    page_title="Whisper AI 转写",
    page_icon="🎧",
    layout="wide"
)

st.markdown("""
<style>
.title {font-size:42px;font-weight:700;}
.subtitle {font-size:18px;color:#666;}
.card {padding:20px;border-radius:15px;border:1px solid #ddd;background:#fafafa;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_whisper_small():
    return WhisperModel(MODEL_DIR, device="cpu", compute_type="int8")

model = load_whisper_small()

os.makedirs(AUDIO_FOLDER, exist_ok=True)
if not os.path.exists(RECORD_FILE):
    with open(RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False)

with st.sidebar:
    st.header("⚙️ 设置")
    st.info("""
模型: Whisper-small

Backend: faster-whisper

Device: CPU INT8

Language: 中文
""")
    st.divider()
    st.header("📚 历史记录")

st.markdown("""
<div class='title'>🎧 Whisper AI</div>
<div class='subtitle'>本地离线语音转文字系统</div>
<br>
🔒 数据不上传服务器　⚡ 快速推理　🌏 中文优化
""", unsafe_allow_html=True)

AUDIO_TYPES = ["mp3", "wav", "m4a", "aac", "flac", "ogg", "opus", "webm", "mp4", "amr", "wma", "aiff"]

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎵 上传音频")
    upload_audio = st.file_uploader("选择音频文件", type=AUDIO_TYPES)

with col2:
    st.subheader("🤖 模型状态")
    st.success("Ready\n\nWhisper-small\n\nCPU INT8")

if upload_audio:
    save_path = os.path.join(AUDIO_FOLDER, upload_audio.name)
    with open(save_path, "wb") as f:
        f.write(upload_audio.getbuffer())

    st.audio(save_path)

    if st.button("🚀 开始转写", use_container_width=True):
        with st.status("正在处理...", expanded=True) as status:
            st.write("🎵 加载音频")
            start = time.time()
            st.write("🤖 Whisper 推理中")
            segments, info = model.transcribe(save_path, language="zh", beam_size=5)
            text_out = "".join(segment.text for segment in segments)
            cost = round(time.time() - start, 2)
            st.write("✅ 生成结果")
            status.update(label="转写完成", state="complete")

        record = {
            "filename": upload_audio.name,
            "audio_path": save_path,
            "trans_text": text_out,
            "cost_time": cost,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(RECORD_FILE, "r", encoding="utf-8") as f:
            records = json.load(f)
        records.append(record)
        with open(RECORD_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        st.subheader("📝 转写结果")
        st.text_area("", text_out, height=250)
        st.success(f"耗时 {cost} 秒")
        st.download_button("📥 下载 TXT", text_out, file_name="result.txt")

st.divider()
st.subheader("📜 历史转写记录")

with open(RECORD_FILE, "r", encoding="utf-8") as f:
    history = json.load(f)

for item in reversed(history):
    with st.expander(f"[{item['create_time']}] {item['filename']} ({item['cost_time']}s)"):
        st.write(item["trans_text"])

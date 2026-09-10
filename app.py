import streamlit as st
import json
import os
import time
from datetime import datetime
from faster_whisper import WhisperModel

# ===== 项目内隔离路径：模型、音频、记录都在项目文件夹内 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "faster-whisper-small")

# set_page_config 必须是第一条 Streamlit 命令
st.set_page_config(page_title="音频转写工具", layout="wide")


@st.cache_resource
def load_whisper_small():
    # 只加载一次模型，避免重复占用内存；CPU 使用 int8，速度与内存占用更均衡
    return WhisperModel(MODEL_DIR, device="cpu", compute_type="int8")


model = load_whisper_small()

AUDIO_FOLDER = os.path.join(BASE_DIR, "audios")
RECORD_FILE = os.path.join(BASE_DIR, "records.json")

# 音频保存目录与记录文件初始化
os.makedirs(AUDIO_FOLDER, exist_ok=True)
if not os.path.exists(RECORD_FILE):
    with open(RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False)

st.title("🎙️基于Whisper的本地音频转写应用")

# 支持常见音频与含音轨的视频格式（底层 PyAV/FFmpeg 解码）
AUDIO_TYPES = ["mp3", "wav", "m4a", "aac", "flac", "ogg", "opus",
               "webm", "mp4", "amr", "wma", "aiff"]
upload_audio = st.file_uploader("上传音频文件", type=AUDIO_TYPES)

if upload_audio is not None:
    # 1. 保存上传音频到本地 audios 文件夹
    save_path = os.path.join(AUDIO_FOLDER, upload_audio.name)
    with open(save_path, "wb") as f:
        f.write(upload_audio.read())

    # 2. 调用 faster-whisper(small) 转写，强制中文
    st.info("正在转写，请稍候，CPU模式速度取决于音频时长……")
    start = time.time()
    segments, info = model.transcribe(save_path, language="zh", beam_size=5)
    # segments 是生成器，需要遍历才会真正执行推理；拼接所有片段文本
    text_out = "".join(segment.text for segment in segments)
    end = time.time()

    cost = round(end - start, 2)

    # 3. 转写记录写入 records.json
    new_record = {
        "filename": upload_audio.name,
        "audio_path": save_path,
        "trans_text": text_out,
        "cost_time": cost,
        "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(RECORD_FILE, "r", encoding="utf-8") as fr:
        records = json.load(fr)
    records.append(new_record)
    with open(RECORD_FILE, "w", encoding="utf-8") as fw:
        json.dump(records, fw, ensure_ascii=False, indent=2)

    # 4. 页面展示结果
    st.audio(save_path)
    st.subheader("转写结果")
    st.text_area("识别文本", text_out, height=220)
    st.success(f"转写完成，耗时 {cost} 秒")

# 5. 历史转写记录
st.divider()
st.subheader("📜历史转写记录")
with open(RECORD_FILE, "r", encoding="utf-8") as f:
    history = json.load(f)

for item in reversed(history):
    with st.expander(f"[{item['create_time']}] {item['filename']} 耗时:{item['cost_time']}s"):
        st.text(item["trans_text"])

import streamlit as st
import json
import os
import time
from datetime import datetime
from faster_whisper import WhisperModel
from ai_summary.summary import AISummary

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "faster-whisper-small")
AUDIO_FOLDER = os.path.join(BASE_DIR, "audios")
RECORD_FILE = os.path.join(BASE_DIR, "records.json")

st.set_page_config(page_title="Whisper AI 转写", page_icon="🎧", layout="wide")

@st.cache_resource
def load_model():
    return WhisperModel(MODEL_DIR, device="cpu", compute_type="int8")

@st.cache_resource
def load_summary_model():
    return AISummary()

model = load_model()
summary_model = load_summary_model()
os.makedirs(AUDIO_FOLDER, exist_ok=True)

if not os.path.exists(RECORD_FILE):
    with open(RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False)

st.markdown("""
<style>
.title{font-size:42px;font-weight:700;}
.subtitle{color:#666;font-size:18px;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 模型信息")
    st.info("Whisper-small\nfaster-whisper\nCPU INT8\n中文识别")

st.markdown("""
<div class='title'>🎧 Whisper AI</div>
<div class='subtitle'>本地离线语音转文字系统</div>
<br>🔒 Offline　⚡ Faster Whisper　🤖 AI摘要
""", unsafe_allow_html=True)

AUDIO_TYPES=["mp3","wav","m4a","aac","flac","ogg","opus","webm","mp4","amr","wma","aiff"]
upload_audio=st.file_uploader("🎵 上传音频文件", type=AUDIO_TYPES)


def format_time(seconds):
    ms=int((seconds-int(seconds))*1000)
    s=int(seconds)%60
    m=int(seconds)//60
    h=m//60
    m=m%60
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def create_srt(segments):
    lines=[]
    for i,seg in enumerate(segments,1):
        lines.append(str(i))
        lines.append(f"{format_time(seg.start)} --> {format_time(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)

if upload_audio:
    save_path=os.path.join(AUDIO_FOLDER, upload_audio.name)
    with open(save_path,"wb") as f:
        f.write(upload_audio.getbuffer())

    st.audio(save_path)

    if st.button("🚀 开始转写", use_container_width=True):
        with st.status("Whisper处理中...") as status:
            start=time.time()
            segments,info=model.transcribe(save_path, language="zh", beam_size=5)
            segment_list=list(segments)
            text_out="".join(s.text for s in segment_list)
            srt_text=create_srt(segment_list)
            cost=round(time.time()-start,2)
            status.update(label="转写完成", state="complete")

        record={
            "filename":upload_audio.name,
            "audio_path":save_path,
            "trans_text":text_out,
            "cost_time":cost,
            "create_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(RECORD_FILE,"r",encoding="utf-8") as f:
            records=json.load(f)
        records.append(record)
        with open(RECORD_FILE,"w",encoding="utf-8") as f:
            json.dump(records,f,ensure_ascii=False,indent=2)

        st.subheader("📝 转写结果")
        st.text_area("",text_out,height=220)

        st.subheader("🤖 AI摘要")
        if st.button("生成摘要"):
            with st.spinner("AI正在整理内容..."):
                summary = summary_model.generate(text_out)
            st.markdown(summary)

        c1,c2=st.columns(2)
        with c1:
            st.download_button("📥 下载 TXT",text_out,file_name="result.txt")
        with c2:
            st.download_button("🎬 下载 SRT 字幕",srt_text,file_name="subtitle.srt")

        st.subheader("⏱️ 时间戳")
        for seg in segment_list:
            st.write(f"{format_time(seg.start)} - {format_time(seg.end)} : {seg.text}")

st.divider()
st.subheader("📜 历史记录")

with open(RECORD_FILE,"r",encoding="utf-8") as f:
    history=json.load(f)

for item in reversed(history):
    with st.expander(f"{item['filename']} | {item['cost_time']}s"):
        st.write(item["trans_text"])

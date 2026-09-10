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

st.markdown("""
<style>
.title{font-size:42px;font-weight:700;}
.subtitle{color:#666;font-size:18px;}
.card{padding:18px;border-radius:12px;background:#f7f7f7;margin-bottom:15px;}
</style>
""", unsafe_allow_html=True)

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

with open(RECORD_FILE,"r",encoding="utf-8") as f:
    history=json.load(f)

with st.sidebar:
    st.header("⚙️ 设置")
    st.info("Whisper-small\nfaster-whisper\nCPU INT8\nAI摘要增强")

st.markdown("<div class='title'>🎧 Whisper AI Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>本地语音转写与智能摘要系统</div>", unsafe_allow_html=True)

c1,c2,c3=st.columns(3)
c1.metric("📁 已处理文件", len(history))
c2.metric("🤖 AI能力", "摘要生成")
c3.metric("🔒 部署方式", "Offline")

st.divider()

st.subheader("🎵 上传音频")

AUDIO_TYPES=["mp3","wav","m4a","aac","flac","ogg","opus","webm","mp4","amr","wma","aiff"]
upload_audio=st.file_uploader("选择音频文件", type=AUDIO_TYPES)


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

        history.append(record)
        with open(RECORD_FILE,"w",encoding="utf-8") as f:
            json.dump(history,f,ensure_ascii=False,indent=2)

        tab1,tab2,tab3=st.tabs(["📝 转写文本","🎬 字幕","🤖 AI摘要"])

        with tab1:
            st.text_area("",text_out,height=220)
            st.download_button("📥 下载TXT",text_out,file_name="result.txt")

        with tab2:
            st.text_area("",srt_text,height=220)
            st.download_button("🎬 下载SRT",srt_text,file_name="subtitle.srt")

        with tab3:
            if st.button("生成摘要"):
                with st.spinner("AI整理中..."):
                    summary=summary_model.generate(text_out)
                st.markdown(summary)

        st.subheader("⏱️ 时间戳")
        for seg in segment_list:
            st.write(f"{format_time(seg.start)} - {format_time(seg.end)} : {seg.text}")

st.divider()
st.subheader("📜 历史记录")

for item in reversed(history):
    with st.expander(f"📄 {item['filename']} | ⏱ {item['cost_time']}s"):
        st.write(item["trans_text"])

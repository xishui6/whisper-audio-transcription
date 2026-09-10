import streamlit as st
import json
import os
import time
from datetime import datetime
from faster_whisper import WhisperModel
from ai_summary.summary import AISummary
from ai_summary.punctuation import TextFormatter

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
MODEL_DIR=os.path.join(BASE_DIR,"models","faster-whisper-small")
AUDIO_FOLDER=os.path.join(BASE_DIR,"audios")
RECORD_FILE=os.path.join(BASE_DIR,"records.json")

st.set_page_config(page_title="Whisper AI Assistant",page_icon="🎧",layout="wide")

@st.cache_resource
def load_model():
    return WhisperModel(MODEL_DIR,device="cpu",compute_type="int8")

@st.cache_resource
def load_summary():
    return AISummary()

@st.cache_resource
def load_formatter():
    return TextFormatter()

model=load_model()
summary_model=load_summary()
formatter=load_formatter()

os.makedirs(AUDIO_FOLDER,exist_ok=True)
if not os.path.exists(RECORD_FILE):
    json.dump([],open(RECORD_FILE,"w",encoding="utf-8"),ensure_ascii=False)

history=json.load(open(RECORD_FILE,"r",encoding="utf-8"))

st.title("🎧 Whisper AI Assistant")
st.caption("语音转写 · 文本优化 · AI摘要")

c1,c2,c3=st.columns(3)
c1.metric("📁历史记录",len(history))
c2.metric("🤖AI能力","文本整理")
c3.metric("🔒模式","Offline ASR")

upload=st.file_uploader("上传音频",type=["mp3","wav","m4a","flac","mp4"])

if upload:
    path=os.path.join(AUDIO_FOLDER,upload.name)
    open(path,"wb").write(upload.getbuffer())
    st.audio(path)

    if st.button("🚀开始转写"):
        start=time.time()
        segments,info=model.transcribe(path,language="zh",beam_size=5)
        segment_list=list(segments)
        raw_text="".join(s.text for s in segment_list)

        with st.spinner("AI正在整理文本..."):
            formatted_text=formatter.format_text(raw_text)

        cost=round(time.time()-start,2)

        record={
            "filename":upload.name,
            "raw_text":raw_text,
            "formatted_text":formatted_text,
            "cost_time":cost,
            "create_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        history.append(record)
        json.dump(history,open(RECORD_FILE,"w",encoding="utf-8"),ensure_ascii=False,indent=2)

        tab1,tab2,tab3=st.tabs(["📝智能整理文本","🎤原始转写","🤖AI摘要"])

        with tab1:
            st.text_area("",formatted_text,height=250)
            st.download_button("下载整理文本",formatted_text,file_name="formatted.txt")

        with tab2:
            st.text_area("",raw_text,height=250)

        with tab3:
            if st.button("生成摘要"):
                st.write(summary_model.generate(formatted_text))

        st.success(f"完成，耗时 {cost}s")

st.divider()
st.subheader("📜历史记录")
for item in reversed(history):
    with st.expander(item["filename"]):
        st.write(item.get("formatted_text",item.get("trans_text","")))

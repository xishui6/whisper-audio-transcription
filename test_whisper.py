# 端到端测试脚本：加载 faster-whisper small 模型并转写指定音频
# 用法：venv\Scripts\python test_whisper.py audios\test.wav
import os
import sys
import time
from faster_whisper import WhisperModel

# 项目内隔离路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "faster-whisper-small")

audio_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, "audios", "test.wav")
if not os.path.isabs(audio_path):
    audio_path = os.path.join(BASE_DIR, audio_path)

print("正在加载 faster-whisper small 模型 ...")
model = WhisperModel(MODEL_DIR, device="cpu", compute_type="int8")

print(f"开始转写: {audio_path}")
t0 = time.time()
segments, info = model.transcribe(audio_path, language="zh", beam_size=5)
texts = []
for segment in segments:
    texts.append(segment.text)
cost = round(time.time() - t0, 2)

print("识别文本:", "".join(texts))
print(f"耗时: {cost} 秒")

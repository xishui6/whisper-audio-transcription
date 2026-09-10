# 模型方案变更说明（供实习报告参考）

## 1. 变更原因
OpenAI 官方模型 CDN（openaipublic.azureedge.net）目前对全部 whisper 模型文件返回 404（tiny/base/small 均不可下载，GitHub 官方仓库也不含权重），openai-whisper 无法获取 small.pt。
因此推理后端由 openai-whisper(PyTorch) 更换为 **faster-whisper(CTranslate2)**，模型权重仍是同一个 Whisper-small 多语言模型，仅推理引擎不同。

## 2. 模型获取
- 来源：ModelScope `Systran/faster-whisper-small`（国内可直连）
- 位置：项目内 `models/faster-whisper-small/`（model.bin 约 461MB，含 config/tokenizer/vocabulary）
- 离线可用，不写用户缓存目录

## 3. 报告中“核心 API”一节需要同步修改的内容

原写法（openai-whisper）：
```python
import whisper
model = whisper.load_model("small")
result = model.transcribe(audio_path, language="zh")
text = result["text"]
```
现写法（faster-whisper）：
```python
from faster_whisper import WhisperModel
model = WhisperModel(model_dir, device="cpu", compute_type="int8")
segments, info = model.transcribe(audio_path, language="zh", beam_size=5)
text = "".join(segment.text for segment in segments)
```
要点：
1. `WhisperModel(目录, device="cpu", compute_type="int8")` 加载本地模型；
2. `transcribe` 返回 `(segments, info)`，**segments 是生成器，遍历时才真正执行推理**；
3. 最终文本由各片段 `segment.text` 拼接得到；
4. 仍用 `language="zh"` 强制中文，避免语种误判。

## 4. 不需要改动的部分
- 总体流程：上传音频 → 保存 audios → 模型推理 → 写 records.json → 页面展示与历史浏览；
- 四大模块划分（上传 / 推理 / 存储 / 历史展示）不变；
- Streamlit 界面、JSON 持久化、`@st.cache_resource` 只加载一次的思路不变。

## 5. 依赖清单变化
- 保留：streamlit==1.38.0
- 新增：faster-whisper==1.2.1（自带 ctranslate2、PyAV 音频解码）
- 移除：openai-whisper、torch、ffmpeg 二进制（PyAV 内置解码，不再需要系统 ffmpeg）

## 6. 测试结果
约 10 秒清晰普通话音频（“大家好，欢迎使用语音转写工具……”），CPU int8 转写耗时 3.56 秒，识别文本与原文一致。

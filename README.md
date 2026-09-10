# 基于 Whisper（faster-whisper）的本地音频转写应用

完全本地离线运行的语音转文字（ASR）应用：上传音频 → 本地模型转写 → 页面展示结果，并把每次转写写入历史记录。音频不经过任何网络接口。

## 功能特性

- 支持 mp3 / wav / m4a / aac / flac / ogg / opus / webm / mp4 / amr / wma / aiff 共 12 种格式（底层 PyAV/FFmpeg 解码）
- 基于 faster-whisper（CTranslate2 推理后端），Whisper-small 多语言模型，CPU int8 即可运行
- 强制中文识别（language="zh"），减少语种误判
- 转写记录 JSON 持久化，支持历史记录浏览
- 模型只加载一次（@st.cache_resource），降低内存占用
- 环境完全隔离，不污染系统

## 目录结构

```
whisper-audio-transcription/
├─ app.py                  # 主程序：上传转写 + JSON 历史记录
├─ test_whisper.py         # 端到端测试脚本
├─ requirements.txt        # 依赖清单
├─ 启动应用.bat             # Windows 双击启动
├─ 报告API变更说明.md       # 实习报告参考（模型方案变更说明）
├─ models/                 # 模型目录（需自行下载，见下文）
├─ audios/                 # 上传音频保存目录（自动创建）
└─ records.json            # 转写历史记录（首次运行自动生成）
```

> `venv/`、`python310/`、`models/`、`ffmpeg/`、`audios/`、`records.json` 已在 `.gitignore` 中排除，不会上传到仓库。

## 环境搭建

### 1. 安装 Python 3.10
建议 Python 3.10.x（高版本可能存在依赖冲突）。

### 2. 创建虚拟环境并安装依赖
```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 下载模型
模型文件约 461MB，从 ModelScope 下载（国内可直连）：
- 地址：https://modelscope.cn/models/Systran/faster-whisper-small/files
- 下载 `config.json`、`model.bin`、`tokenizer.json`、`vocabulary.txt` 四个文件
- 放到项目目录的 `models/faster-whisper-small/` 下

## 运行

方式一（Windows）：双击 `启动应用.bat`

方式二（命令行）：
```bat
venv\Scripts\python.exe -m streamlit run app.py
```
浏览器打开 http://localhost:8501

## 自测

```bat
venv\Scripts\python.exe test_whisper.py audios\test.wav
```
实测：约 10 秒清晰普通话音频，CPU int8 转写耗时约 3.6 秒，文本准确。

## 技术栈

| 组件 | 版本/说明 |
|---|---|
| Python | 3.10.x |
| faster-whisper | 1.2.1，CTranslate2 推理后端，CPU int8 |
| ctranslate2 | 4.8.2 |
| streamlit | 1.38.0 |
| PyAV(av) | 17.1.0，内置音频解码，无需系统安装 ffmpeg |
| 模型 | faster-whisper-small（与 OpenAI Whisper-small 同权重） |

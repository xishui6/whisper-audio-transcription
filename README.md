# 基于 Whisper（faster-whisper）的本地音频转写应用

完全本地离线运行的语音转文字（ASR）应用：上传音频 → 本地模型转写 → 自动断句加标点 → 说话人分离换行 → 页面展示结果，并把每次转写写入历史记录。音频不经过任何网络接口。

## 功能特性

- 支持 mp3 / wav / m4a / aac / flac / ogg / opus / webm / mp4 / amr / wma / aiff 共 12 种格式（底层 PyAV/FFmpeg 解码）
- 基于 faster-whisper（CTranslate2 推理后端），Whisper-small 多语言模型，CPU int8 即可运行
- 强制中文识别（language="zh"），减少语种误判
- **标点断句**：本地规则引擎（RulePunctuator）利用词级时间戳自动断句，疑问句补「？」、感叹句补「！」、语气词/连接词断句、长句按停顿时长兜底，无需联网
- **说话人分离（可选开关）**：本地 CAM++ 说话人分离模型（ModelScope，国内直连下载），把不同说话人的话按人换行展示（【说话人 N】）；失败自动降级为按停顿分段
- **AI 标点精修（可选）**：配置 OpenAI 兼容接口后，用 LLM 进一步修正标点与口语错字，网络不通时自动回退本地规则结果
- 转写记录 JSON 持久化，支持历史记录浏览
- 模型只加载一次（@st.cache_resource），降低内存占用
- 环境完全隔离，不污染系统

## 目录结构

```
whisper-audio-transcription/
├─ app.py                  # 主程序：上传转写 + 标点/说话人处理 + JSON 历史记录
├─ ai_summary/             # 功能模块
│  ├─ config.py            # 读取 .env / 环境变量（API key 等）
│  ├─ punctuation.py       # 本地规则标点引擎（词级状态机）
│  ├─ diarize.py           # 说话人分离封装（CAM++，本地模型）
│  ├─ format_utils.py      # 文本组装（按说话人换行）+ LLM 精修（含回退）
│  ├─ summary.py           # AI 摘要（可选在线功能）
│  └─ prompt.py            # 摘要 prompt
├─ test_whisper.py         # 端到端测试脚本
├─ requirements.txt        # 依赖清单
├─ 启动应用.bat             # Windows 双击启动
├─ 报告API变更说明.md       # 实习报告参考（模型方案变更说明）
├─ models/                 # 模型目录（faster-whisper + modelscope 说话人模型）
├─ audios/                 # 上传音频保存目录（自动创建）
└─ records.json            # 转写历史记录（首次运行自动生成）
```

> `venv/`、`python310/`、`models/`、`ffmpeg/`、`audios/`、`records.json`、`.env` 已在 `.gitignore` 中排除，不会上传到仓库。

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

## AI 功能配置（可选）

「AI 标点精修」与「AI 摘要」为在线功能（OpenAI 兼容接口），不配置也不影响本地转写与标点/说话人分段。

推荐方式：在项目根目录创建 `.env` 文件（已加入 .gitignore），**直接支持 DeepSeek 等 OpenAI 兼容接口**：

```env
OPENAI_API_KEY=你的API密钥
OPENAI_BASE_URL=https://api.deepseek.com   # DeepSeek 官方接口；用中转/其他服务时改成对应地址
SUMMARY_MODEL=deepseek-chat                 # DeepSeek-V3；其他服务填对应模型名
```

也可以在启动前用系统环境变量设置（效果相同）。

- 未配置 `OPENAI_API_KEY`：标点走本地规则引擎，说话人分段照常，侧边栏给出提示。
- 已配置但网络不通：自动回退本地规则结果，不影响出结果，只是稍等几秒。
- 说话人分离模型（CAM++，约 400MB）首次运行自动从 ModelScope 下载（国内直连、无需 token），之后缓存于 `models/modelscope/`，CPU 上约需 30~60 秒。

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

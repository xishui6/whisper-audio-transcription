# -*- coding: utf-8 -*-
"""本地说话人分离（diarization）封装。

基于 ModelScope 的 CAM++ 说话人分离模型（iic/speech_campplus_speaker-diarization_common），
模型从 ModelScope 下载（国内可直连，无需 token），CPU 即可运行。
首次运行会自动下载模型（约 200MB）；模型不可用或执行失败时优雅返回 None。
"""

import os
import re
import subprocess
import tempfile

# 项目内置 ffmpeg（用于把任意音频统一转为 16k 单声道 wav 喂给分离模型）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFMPEG = os.path.join(BASE_DIR, "ffmpeg", "bin", "ffmpeg.exe")

# 模型缓存与 SDK 配置目录都放到项目内 models/ 下，避免写入用户目录、方便整体迁移
MODELSCOPE_CACHE = os.path.join(BASE_DIR, "models", "modelscope")
MODELSCOPE_HOME = os.path.join(BASE_DIR, "models", "modelscope_home")
os.environ.setdefault("MODELSCOPE_CACHE", MODELSCOPE_CACHE)
os.environ.setdefault("MODELSCOPE_HOME", MODELSCOPE_HOME)

MODEL_ID = "iic/speech_campplus_speaker-diarization_common"
MODEL_REVISION = "master"


class SpeakerDiarizer:
    """说话人分离器：audio_path -> [(start, end, speaker_id), ...]"""

    def __init__(self):
        self._pipeline = None
        self._error = None

    # ---------- 内部工具 ----------

    def _load(self):
        """懒加载 modelscope pipeline（首次调用触发模型下载）。"""
        if self._pipeline is None and self._error is None:
            try:
                from modelscope.pipelines import pipeline
                from modelscope.utils.constant import Tasks
                self._pipeline = pipeline(
                    task=Tasks.speaker_diarization,
                    model=MODEL_ID,
                    model_revision=MODEL_REVISION,
                )
            except Exception as e:
                self._error = f"diarization 模型加载失败: {e!r}"
        return self._pipeline

    def _to_wav16k(self, audio_path):
        """用项目内 ffmpeg 把任意格式音频转为 16k 单声道 wav。"""
        if not os.path.exists(FFMPEG):
            self._error = "未找到 ffmpeg.exe"
            return None
        base = os.path.splitext(os.path.basename(audio_path))[0]
        tmp_wav = os.path.join(tempfile.gettempdir(), f"whisper_diar_{base}.wav")
        cmd = [FFMPEG, "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", tmp_wav]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=180)
        except Exception as e:
            self._error = f"ffmpeg 执行失败: {e!r}"
            return None
        if proc.returncode != 0 or not os.path.exists(tmp_wav):
            self._error = "ffmpeg 转码失败: " + proc.stderr.decode("utf-8", "ignore")[-300:]
            return None
        return tmp_wav

    @staticmethod
    def _parse(result):
        """解析 modelscope speaker-diarization 输出 → [(start, end, spk_id), ...]

        兼容两种输出：{'text': [[start, end, spk], ...]} 或 {'text': [{'start','end','spk'}, ...]}
        """
        out = []
        text_list = []
        if isinstance(result, dict):
            text_list = result.get("text") or []
        elif isinstance(result, list):
            text_list = result
        for item in text_list:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                try:
                    spk = str(item[2])
                    m = re.search(r"(\d+)", spk)
                    spk_id = int(m.group(1)) if m else 0
                    out.append((float(item[0]), float(item[1]), spk_id))
                except (TypeError, ValueError):
                    continue
            elif isinstance(item, dict):
                spk = str(item.get("spk", "spk0"))
                m = re.search(r"(\d+)", spk)
                spk_id = int(m.group(1)) if m else 0
                try:
                    out.append((float(item.get("start", 0.0)),
                                float(item.get("end", 0.0)),
                                spk_id))
                except (TypeError, ValueError):
                    continue
        out.sort(key=lambda x: x[0])
        # 说话人标签重映射为连续编号 0,1,2...（聚类输出可能不连续）
        order = {}
        nxt = 0
        for i, (s, e, spk) in enumerate(out):
            if spk not in order:
                order[spk] = nxt
                nxt += 1
            out[i] = (s, e, order[spk])
        return out

    # ---------- 对外接口 ----------

    def diarize(self, audio_path):
        """返回 [(start, end, speaker_id), ...]；模型不可用 / 失败返回 None。"""
        pipe = self._load()
        if pipe is None:
            return None
        wav = self._to_wav16k(audio_path)
        if wav is None:
            return None
        try:
            result = pipe(wav)
            return self._parse(result)
        except Exception as e:
            self._error = f"diarization 推理失败: {e!r}"
            return None
        finally:
            # 清理临时 wav
            try:
                if wav and os.path.exists(wav):
                    os.remove(wav)
            except OSError:
                pass

    @property
    def error(self):
        return self._error

import os
from pathlib import Path

# 项目根目录下的 .env（已被 .gitignore 排除），不存在则忽略
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def _load_env_file():
    if not _ENV_FILE.exists():
        return
    try:
        for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())
    except OSError:
        pass


_load_env_file()

API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("SUMMARY_MODEL", "gpt-4o-mini")

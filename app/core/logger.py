"""结构化日志：JSON 格式，含 trace_id / 时间戳 / 级别。

生产落地考虑：
  - JSON 格式便于 ELK / Loki / 云日志平台采集与检索
  - 每个请求通过 X-Trace-Id 透传 trace_id，便于链路追踪（结合 contextvars）
"""
import json
import logging
import os
import sys
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler

_configed = False

# 全局 trace_id（由中间件或请求入口设置，日志自动附带）
_trace_id: ContextVar[str] = ContextVar("trace_id", default="")

# 额外透传字段（如 patient_id）
_extras: ContextVar[dict] = ContextVar("log_extras", default={})


class JsonFormatter(logging.Formatter):
    """把日志格式化为单行 JSON。"""

    def format(self, record: logging.LogRecord) -> str:
        data = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": _trace_id.get(),
        }
        data.update(_extras.get())
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        # 兼容 %s|key=value 风格之外的额外属性
        for key, value in record.__dict__.items():
            if key not in {"name", "msg", "args", "levelname", "levelno",
                           "pathname", "filename", "module", "exc_info", "exc_text",
                           "stack_info", "lineno", "funcName", "created", "msecs",
                           "relativeCreated", "thread", "threadName", "processName",
                           "process", "taskName", "message"} and not key.startswith("_"):
                data[key] = value
        return json.dumps(data, ensure_ascii=False, default=str)


def set_trace_id(trace_id: str) -> None:
    _trace_id.set(trace_id)


def set_extra(key: str, value) -> None:
    _extras.set({**_extras.get(), key: value})


def configure_logging(level: str = "INFO", log_dir: str = "./logs",
                      log_file: str = "agent_service.log") -> None:
    global _configed
    if _configed:
        return
    formatter = JsonFormatter()
    handlers: list[logging.Handler] = []

    # ① 控制台输出：本地/容器内实时查看
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    handlers.append(console)

    # ② 文件输出：落盘到项目 logs/ 目录，按大小轮转，避免单文件无限增长
    try:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, log_file),
            maxBytes=10 * 1024 * 1024,   # 单文件 10MB
            backupCount=5,               # 最多保留 5 个历史文件
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    except Exception:
        pass  # 落盘失败不影响控制台日志

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers = handlers
    _configed = True


def get_logger(name: str = "agent_service") -> logging.Logger:
    return logging.getLogger(name)
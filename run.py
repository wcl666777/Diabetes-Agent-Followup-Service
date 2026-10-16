"""Agent 服务启动脚本。

在 agent_service 目录下执行（也可直接在 IDE 中运行本文件）：
    python run.py

等价于命令行：
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
import uvicorn

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    host = settings.app_host
    port = settings.app_port
    # 0.0.0.0 无法直接点击访问，浏览器统一用 127.0.0.1
    browse_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    base = f"http://{browse_host}:{port}"

    print("=" * 56)
    print("  Agent 服务已启动，可通过以下地址访问：")
    print(f"    前端页面 : {base}/")
    print(f"    接口文档 : {base}/docs")
    print(f"    健康检查 : {base}/health")
    print("=" * 56, flush=True)

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=False,  # 开发时可改为 True 以自动重载
    )


if __name__ == "__main__":
    main()

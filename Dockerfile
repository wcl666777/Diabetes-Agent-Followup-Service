FROM python:3.11-slim

# 设置为非交互安装 & 中文编码
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /app

# 先装依赖（利用镜像层缓存，代码变动不会触发重新下载）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 拷贝源码与静态资源
COPY app ./app
COPY static ./static
COPY scripts ./scripts

EXPOSE 8000

# 启动 Agent 服务（uvicorn 生产模式：异步 + 多 worker 由外部编排控制）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
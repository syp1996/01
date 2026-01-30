# ==========================================
# 第一阶段：构建器 (Builder Stage)
# ==========================================
# 使用 bookworm 稳定版，避免 trixie (测试版) 的潜在不兼容
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 安装编译依赖 (gcc 和 libpq-dev)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 构建 Wheels
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt


# ==========================================
# 第二阶段：运行时 (Runtime Stage)
# ==========================================
FROM python:3.11-slim-bookworm

WORKDIR /app

# 【关键修改】修正包名为 libpq5
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# 从第一阶段复制构建产物
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

# 安装依赖
RUN pip install --no-cache /wheels/*

# 复制代码
COPY . .

# 创建非 root 用户
RUN addgroup --system appgroup && adduser --system --group appuser

# 【关键修复】将 /app 目录的所有权移交给 appuser，否则无法写入日志
RUN chown -R appuser:appgroup /app

# 切换用户
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
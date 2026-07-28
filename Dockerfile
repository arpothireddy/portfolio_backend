FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py persona.py ./

# Cloud Run injects PORT (defaults to 8080). Must bind to it, not a hardcoded port.
ENV PORT=8080
EXPOSE 8080

# Shell form so ${PORT} is expanded at runtime by Cloud Run.
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1

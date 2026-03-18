FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    graphviz \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Generate data and train model during build
RUN python data/generate_data.py && python src/model_trainer.py

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Supervisor configuration for running both services
RUN echo '[supervisord]\n\
nodaemon=true\n\
\n\
[program:fastapi]\n\
command=uvicorn api.main:app --host 0.0.0.0 --port 8000\n\
autostart=true\n\
autorestart=true\n\
stdout_logfile=/dev/stdout\n\
stdout_logfile_maxbytes=0\n\
stderr_logfile=/dev/stderr\n\
stderr_logfile_maxbytes=0\n\
\n\
[program:streamlit]\n\
command=streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true\n\
autostart=true\n\
autorestart=true\n\
stdout_logfile=/dev/stdout\n\
stdout_logfile_maxbytes=0\n\
stderr_logfile=/dev/stderr\n\
stderr_logfile_maxbytes=0' > /etc/supervisor/conf.d/salon.conf

CMD ["supervisord", "-c", "/etc/supervisor/conf.d/salon.conf"]

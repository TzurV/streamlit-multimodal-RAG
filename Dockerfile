# app/Dockerfile

FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

#RUN git clone https://github.com/streamlit/streamlit-example.git .
COPY requirements.txt .
COPY streamlit_app.py .

RUN pip3 install -r requirements.txt

EXPOSE 8001

HEALTHCHECK CMD curl --fail http://localhost:8001/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8001", "--server.address=0.0.0.0"]
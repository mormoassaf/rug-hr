FROM python:3.10

WORKDIR /src

COPY requirements.txt /src/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /src/requirements.txt

COPY app /src/app
COPY scripts /src/scripts

RUN python3.10 ./scripts/download_artifacts.py

CMD ["python3.10", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
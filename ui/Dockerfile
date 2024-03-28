FROM python:3.10

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY gui.py /app
COPY style.css /app

CMD ["python", "gui.py"]

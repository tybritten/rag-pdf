FROM python:3.10

WORKDIR /app


COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY embed.py /app

CMD ["python", "embed.py"]

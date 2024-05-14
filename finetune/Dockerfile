FROM vmtyler/pdk:gui-v0.0.18a

WORKDIR /app

#RUN apt-get update && apt-get install -y tesseract-ocr libtesseract-dev ffmpeg libsm6 libxext6 libmagic-dev libpoppler-dev poppler-utils


RUN pip install llama-index-finetuning==0.1.4 pydantic==1.10.14

COPY generate_qna_pairs.py /app

CMD ["python", "generate_qna_pairs.py"]

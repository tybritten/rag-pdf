FROM python:3.11

WORKDIR /app



RUN pip install -U pip
RUN pip install llama-index-vector-stores-chroma==0.1.8 llama-index-llms-openai-like==0.1.3 llama-index-embeddings-openai==0.1.11 llama-index-llms-openai==0.1.29
RUN pip install streamlit==1.36.0 chromadb==0.5.5 transformers==4.42.4 
RUN pip install jwt iso-639 loguru
RUN python3 -m nltk.downloader punkt

COPY gui.py /app
COPY static/ /app/static
COPY .streamlit /app/.streamlit

CMD ["python", "gui.py"]

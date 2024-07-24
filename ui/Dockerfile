FROM python:3.11

WORKDIR /app



RUN pip install -U pip
RUN pip install llama-index-llms-openllm==0.1.4
RUN pip install llama-index-llms-huggingface llama-index-vector-stores-chroma==0.1.8 llama-index-embeddings-huggingface llama-index-embeddings-openai 
RUN pip install streamlit chromadb transformers


COPY gui.py /app
COPY static/ /app/static
COPY .streamlit /app/.streamlit

CMD ["python", "gui.py"]

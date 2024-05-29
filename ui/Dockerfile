FROM python:3.11

WORKDIR /app

#COPY requirements.txt ./

RUN pip install -U pip
RUN pip install llama-index-llms-openllm 
RUN pip install llama-index-llms-huggingface llama-index-vector-stores-chroma llama-index-embeddings-huggingface
RUN pip install streamlit chromadb
RUN pip install auto-gptq
RUN  pip install git+https://github.com/huggingface/optimum.git -U
RUN pip install git+https://github.com/huggingface/transformers.git -U
RUN pip install --upgrade accelerate

COPY gui.py /app
COPY static/ /app/static
COPY .streamlit /app/.streamlit

CMD ["python", "gui.py"]

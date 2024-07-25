import argparse
import os

import chromadb
import streamlit as st
from llama_index.core import (Settings, VectorStoreIndex,
                              get_response_synthesizer)
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores.types import (MetadataFilter,
                                                  MetadataFilters)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openllm import OpenLLMAPI
from llama_index.vector_stores.chroma import ChromaVectorStore
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--path-to-db", type=str, default="db", help="path to chroma db")
parser.add_argument(
    "--emb-model-path",
    type=str,
    default=None,
    help="local path or URL to sentence transformer model",
)
parser.add_argument(
    "--path-to-chat-model",
    default="TinyLlama/TinyLlama-1.1B-Chat-v0.4",
    help="local path or URL to chat model",
)
parser.add_argument(
    "--top-k",
    default=5,
    type=int,
    help="top k results",
)
parser.add_argument(
    "--cutoff",
    default=0.7,
    type=float,
    help="cutoff for similarity score",
)
parser.add_argument(
    "--streaming",
    default=True,
    help="stream responses",
    action=argparse.BooleanOptionalAction,
)
args = parser.parse_args()

st.set_page_config(
    layout="wide", page_title="Retrieval Augmented Generation (RAG) Demo Q&A"
)

with open("static/style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)


######

# CSS for formatting top bar
st.markdown(
    """
    <style>
    .top-bar {
        background-color: #00B188;
        padding: 15px;
        color: white;
        margin-top: -82px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create top bar
st.markdown(
    """
    <div class="top-bar">
         <img src="/app/static/hpe_pri_wht_rev_rgb.png" alt="HPE Logo" height="55">  
    </div>
    """,
    unsafe_allow_html=True,
)

######

st.header("Retrieval Augmented Generation (RAG) Demo Q&A", divider="gray")

st.session_state.temp = 0.2
st.session_state.top_p = 0.8
st.session_state.max_length = 250
st.session_state.cutoff = args.cutoff
st.session_state.top_k = args.top_k


@st.cache_data
def load_chat_model(
    cuda_device="cuda:0",
    temp=st.session_state.temp,
    max_length=st.session_state.max_length,
    top_p=st.session_state.top_p,
):
    generate_kwargs = {
        "do_sample": True,
        "temperature": temp,
        "top_p": top_p,
        "max_length": max_length,
    }
    if args.path_to_chat_model.startswith("http"):
        st.write(f"Using OpenLLM model endpoint: {args.path_to_chat_model}")
        modelpath = str(args.path_to_chat_model)
        llm = OpenLLMAPI(address=modelpath, generate_kwargs=generate_kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.path_to_chat_model)
        stopping_ids = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        llm = HuggingFaceLLM(
            model_name=args.path_to_chat_model,
            tokenizer_name=args.path_to_chat_model,
            generate_kwargs=generate_kwargs,
            stopping_ids=stopping_ids,
        )
        st.write(f"Using local model: {args.path_to_chat_model}")
    Settings.llm = llm
    return None


def load_data():
    if args.emb_model_path.startswith("http"):
        st.write(f"Using Embedding API model endpoint: {args.emb_model_path}")
        embed_model = OpenAIEmbedding(api_base=args.emb_model_path, api_key="dummy")
    else:
        st.write(f"Embedding model: {args.emb_model_path}")
        embed_model = HuggingFaceEmbedding(model_name=args.emb_model_path)
    chroma_client = chromadb.PersistentClient(args.path_to_db)
    chroma_collection = chroma_client.get_collection(name="documents")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )
    return index, chroma_collection.get()


def create_query_engine(
    filters=None, cutoff=st.session_state.cutoff, top_k=st.session_state.top_k
):
    retriever = VectorIndexRetriever(
        index=index, similarity_top_k=top_k, filters=filters
    )
    # configure response synthesizer
    response_synthesizer = get_response_synthesizer(
        response_mode="no_text", streaming=False
    )
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity=cutoff)],
    )
    # query_engine = index.as_query_engine(similarity_top_k=args.top_k, streaming=True)
    return query_engine


welcome_message = "Hello, I am HPE Document chat. \n\n Please ask me any questions related to the documents listed below. If there are no documents listed, please select a tag below to filter."
col1, col2 = st.columns(2)

chat_container = col1.container(height=435, border=False)
input_container = col1.container()


with st.spinner(f"Loading {args.path_to_chat_model} q&a model..."):
    llm = load_chat_model()

with st.spinner(f"Loading data and {args.emb_model_path} embedding model..."):
    index, chunks = load_data()


# uploaded_files = col2.file_uploader("Upload Files", accept_multiple_files=True)
tags = []
uploaded_files = {}
filters = None
for i in range(len(chunks["ids"])):
    file = chunks["metadatas"][i]["Source"]
    eltags = chunks["metadatas"][i]["Tag"]
    if eltags not in tags:
        tags.append(eltags)
    if eltags not in uploaded_files:
        uploaded_files[eltags] = []
    if file not in uploaded_files[eltags]:
        uploaded_files[eltags].append(file)


def list_sources():
    col2.markdown("##### List of Sources:")
    global filters
    filter_tags = st.session_state["tags"] if "tags" in st.session_state else []
    if len(filter_tags) > 0:
        meta_filters = []
        for tag in filter_tags:
            with col2.expander(tag):
                files = uploaded_files[tag]
                for file in files:
                    st.write(file)
            meta_filters.append(MetadataFilter(key="Tag", value=tag))
        filters = MetadataFilters(
            filters=meta_filters,
            condition="or",
        )
    else:
        for tag in uploaded_files:
            with col2.expander(tag):
                files = uploaded_files[tag]
                for file in files:
                    st.write(file)


if len(tags) > 0:
    filter_tags = col2.multiselect(
        "Select Tags to Filter on:", tags, on_change=list_sources(), key="tags"
    )
    col1.divider()
# elif len(tags) == 1:
#    filter_tags = col1.multiselect("Select Tags for Retrieval", tags, default=tags[0], on_change=list_sources, key='tags')


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add welcome message to new chat history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": welcome_message,
            "avatar": "./static/logo.jpeg",
        }
    )

for message in st.session_state.messages:
    if "avatar" not in message:
        message["avatar"] = None
    with chat_container.chat_message(message["role"], avatar=message["avatar"]):
        st.write(message["content"])

default_instructions = "If you don't know the answer to a question, please don't share false information. \n Limit your response to 500 tokens."
brief = "just generate the answer without a lot of explanations."


def reload():
    with st.spinner(f"Loading {args.path_to_chat_model} q&a model..."):
        llm = load_chat_model(
            temp=st.session_state.temp,
            top_p=st.session_state.top_p,
            max_length=st.session_state.max_length,
        )
    global query_engine
    query_engine = create_query_engine(
        cutoff=st.session_state.cutoff, top_k=st.session_state.top_k, filters=filters
    )


def output_stream(llm_stream):
    for chunk in llm_stream:
        yield chunk.delta


with col1.expander("Settings"):
    temp = st.slider("Temperature", 0.0, 1.0, key="temp")
    top_k = st.slider("Top K", 1, 25, key="top_k")
    cutoff = st.slider("Cutoff", 0.0, 1.0, key="cutoff")
    instructions = st.text_area("Prompt Instructions", default_instructions)
    st.button("Update Settings", on_click=reload())

# Accept user input
if prompt := input_container.chat_input("Say something..."):

    with chat_container.chat_message("user"):
        st.write(prompt)

    print(f"Querying with prompt: {prompt}")
    output = query_engine.query(prompt)
    context_str = ""
    for node in output.source_nodes:
        print(f"Context: {node.metadata}")
        context_str += node.text.replace("\n", "  \n")
    text_qa_template_str_llama3 = f"""
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>
        Context information is
        below.
        ---------------------
        {context_str}
        ---------------------
        Using
        the context information, answer the question: {prompt}
        {instructions}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
    llm = Settings.llm
    if args.streaming:
        output_response = llm.stream_complete(
            text_qa_template_str_llama3, formatted=True
        )
        with chat_container.chat_message("assistant", avatar="./static/logo.jpeg"):
            response = st.write_stream(output_stream(output_response))
    else:
        output_response = llm.complete(text_qa_template_str_llama3)
        print(output_response)
        with chat_container.chat_message("assistant", avatar="./static/logo.jpeg"):
            response = st.write(output_response.text)

    project = os.getenv("PPS_PROJECT_NAME", "default")
    doc_repo = os.getenv("DOCUMENT_REPO", "documents")
    proxy_url = os.getenv("PACH_PROXY_EXTERNAL_URL_BASE", "http://localhost:30080")

    with col2:
        references = output.source_nodes
        for i in range(len(references)):
            title = references[i].node.metadata["Source"]
            page = references[i].node.metadata["Page Number"]
            text = references[i].node.text
            commit = references[i].node.metadata["Commit"]
            doctag = references[i].node.metadata["Tag"]
            newtext = text.encode("unicode_escape").decode("unicode_escape")
            out_title = f"**Source:** {title}  \n **Page:** {page}  \n **Similarity Score:** {round((references[i].score * 100),3)}% \n"
            out_text = f"**Text:**  \n {newtext}  \n"
            title = title.replace(" ", "%20")
            if doctag:
                doctag = doctag.replace(" ", "%20")
                out_link = f"[Link to file in Commit {commit}]({proxy_url}/proxyForward/pfs/{project}/{doc_repo}/{commit}/{doctag}/{title}#page={page})\n"
            else:
                out_link = f"[Link to file in Commit {commit}]({proxy_url}/proxyForward/pfs/{project}/{doc_repo}/{commit}/{title}#page={page})\n"
            col2.markdown(out_title)
            col2.write(out_text, unsafe_allow_html=True)
            if not title.startswith("http"):
                col2.write(out_link)
            col2.divider()

import chromadb
import argparse
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
import torch
from llama_index.core import Settings
import streamlit as st
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--path-to-db", type=str, default="db", help="path to csv containing press releases"
)
parser.add_argument(
    "--emb-model-path",
    type=str,
    default=None,
    help="path to locally saved sentence transformer model",
)
parser.add_argument(
    "--path-to-chat-model",
    default="TinyLlama/TinyLlama-1.1B-Chat-v0.4",
    help="path to chat model",
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
    "--response-mode",
    default="tree_summarize",
    type=str,
    help="LLamaIndex Response Mode",
)
args = parser.parse_args()



st.set_page_config(
    layout="wide", page_title="Retrieval Augmented Generation (RAG) Demo Q&A"
)

with open("style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)


######

# CSS for formatting top bar
st.markdown(
    """
    <style>
    .top-bar {
        background-color: #0D5265;
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
         <img src="https://www.xma.co.uk/wp-content/uploads/2018/06/hpe_pri_wht_rev_rgb.png" alt="HPE Logo" height="60">  
    </div>
    """,
    unsafe_allow_html=True,
)

######

st.title("Retrieval Augmented Generation (RAG) Demo Q&A")


@st.cache_resource
def load_chat_model(cuda_device="cuda:0"):
    st.write(f"Chat Model:  {args.path_to_chat_model}")

    tokenizer = AutoTokenizer.from_pretrained(args.path_to_chat_model)

    stopping_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    llm = HuggingFaceLLM(
        model_name=args.path_to_chat_model,
        tokenizer_name=args.path_to_chat_model,
        generate_kwargs={
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.9,
            "max_length": 512
        },
        stopping_ids=stopping_ids,
    )
    Settings.llm = llm
    return llm


@st.cache_resource
def load_data():
    st.write(f"Embedding model: {args.emb_model_path}")
    embed_model = HuggingFaceEmbedding(model_name=args.emb_model_path)
    chroma_client = chromadb.PersistentClient(args.path_to_db)
    chroma_collection = chroma_client.get_collection(name="documents")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )
    return index


@st.cache_resource
def create_query_engine():
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=args.top_k,
    )
    # configure response synthesizer
    response_synthesizer = get_response_synthesizer(response_mode=args.response_mode, streaming=True)
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=args.cutoff)],
    )

    return query_engine


with torch.inference_mode():
    with st.spinner(f"Loading {args.path_to_chat_model} q&a model..."):
        llm = load_chat_model()
        # model_fine_tuned = load_qa_model(cuda_device="cuda:1")

    with st.spinner(f"Loading data and {args.emb_model_path} embedding model..."):
        index = load_data()

    with st.spinner("Creating query engine..."):
        query_engine = create_query_engine()

    user_input = st.text_input(label="What's your question?", key="input")

    if user_input:


        output = query_engine.query(user_input)

        outmark = st.empty()
        response = ""
        for token in output.response_gen:
            response = response + str(token)
            outmark.markdown(response)

        st.divider()
        with st.spinner("Processing..."):
            references = output.get_response().source_nodes
            for i in range(len(references)):
                title = references[i].node.metadata["Source"]
                page = references[i].node.metadata["Page Number"]
                text = references[i].node.text
                commit = references[i].node.metadata["Commit"]
                newtext = text.encode('unicode_escape').decode('unicode_escape').replace('\n', "")
                out_title = f"##### Source: {title} \n Page: {page}\n Similarity Score: {round((references[i].score * 100),3)}% \n"
                out_text = f"##### Text:\n {newtext} \n"
                out_link = f"##### [Link to file in Commit {commit}](http://mldm-pachyderm.us.rdlabs.hpecorp.net/proxyForward/pfs/pdf-rag/documents/{commit}/{title}#page={page})\n"
                st.markdown(out_title)
                st.markdown(out_text)
                if not title.startswith("http"):
                    st.markdown(out_link)
                st.divider()

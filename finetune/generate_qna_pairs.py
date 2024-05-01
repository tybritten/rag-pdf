import json
import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode

from llama_index.core.schema import TextNode
from llama_index.finetuning import generate_qa_embedding_pairs
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset

import chromadb
import argparse
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings
import random

parser = argparse.ArgumentParser()
parser.add_argument(
    "--path-to-db", type=str, default="db", help="path to csv containing press releases"
)
parser.add_argument(
    "--emb-model-path",
    type=str,
    default="BAAI/bge-base-en-v1.5",
    help="path to locally saved sentence transformer model",
)
parser.add_argument(
    "--path-to-chat-model",
    default="TinyLlama/TinyLlama-1.1B-Chat-v0.4",
    help="path to chat model",
)
parser.add_argument("--output", default="./output", help="output directory")

parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Fraction of nodes to use for training set (between 0 and 1). Validation set will be 1-train_ratio. ')
args = parser.parse_args()

def split_nodes(nodes, train_fraction=0.8):
    # Ensure there are enough nodes to form a meaningful split
    assert len(nodes) >= 2, "Node list must contain at least two elements."
    assert 0 < train_fraction < 1, "Train fraction must be between 0 and 1."

    LEN = len(nodes)
    train_size = int(LEN * train_fraction)
    validation_size = LEN - train_size  # Ensures all elements are included
    print("Training set size: ",train_size)
    print("Validation set size: ",validation_size)
    # Shuffle nodes to randomize the sampling
    shuffled_nodes = random.sample(nodes, LEN)

    train = shuffled_nodes[:train_size]
    validation = shuffled_nodes[train_size:]

    return train, validation


llm = HuggingFaceLLM(
    context_window=8192,
    max_new_tokens=512,
    model_name=args.path_to_chat_model,
    tokenizer_name=args.path_to_chat_model,
)
Settings.llm = llm


embed_model = HuggingFaceEmbedding(model_name=args.emb_model_path)
chroma_client = chromadb.PersistentClient(args.path_to_db)
chroma_collection = chroma_client.get_collection(name="documents")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
chunks = chroma_collection.get()

nodes = []
for i in range(len(chunks["ids"])):
    nodes.append(TextNode(id=chunks["ids"][i], text=chunks["documents"][i], metadata=chunks["metadatas"][i]))

print("Number of nodes: ",len(nodes))

train, test = split_nodes(nodes, train_fraction=args.train_ratio)  # 80% train, 20% validation


train_dataset = generate_qa_embedding_pairs(
    llm=llm, nodes=train
)
test_dataset = generate_qa_embedding_pairs(
    llm=llm, nodes=test
)

train_dataset.save_json(f"{args.output}/train_dataset.json")
test_dataset.save_json(f"{args.output}/test_dataset.json")


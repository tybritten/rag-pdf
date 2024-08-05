import argparse
import json
import os
import shutil

import chromadb
import torch
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.storage import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

def main(data_path, embed_model, db):
    collection = db.get_or_create_collection(
        name="documents", metadata={"hnsw:space": "cosine"}
    )
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    docs = []
    index = VectorStoreIndex(
        docs, storage_context=storage_context, embed_model=embed_model
    )
    for dirpath, dirs, files in os.walk(data_path):
        for file in files:
            input_file = os.path.join(dirpath, file)

            with open(input_file, "r") as f:
                input_text = json.load(f)
                for doc in input_text:
                    # (4.26.2024) Andrew: Dealing with issue when parsing txt or xml, doc can be a string
                    if isinstance(doc, dict):
                        if doc["data_type"] == "Table":
                            text = doc["metadata"]["text_as_html"]
                        else:
                            text = doc["content"]
                        source = doc["metadata"]["source"]
                        if "page_number" in doc["metadata"]:
                            page_number = doc["metadata"]["page_number"]
                        else:
                            page_number = 1
                        if "tag" in doc["metadata"]:
                            tag = doc["metadata"]["tag"]
                        else:
                            tag = ""
                        metadata = {
                            "Source": source,
                            "Page Number": page_number,
                            "Commit": os.environ.get("PACH_JOB_ID", ""),
                            "Tag": tag,
                        }
                        docs.append(TextNode(text=text, metadata=metadata))

    print("Number of chunks: ", len(docs))

    index.insert_nodes(docs, show_progress=True)
    print("Indexing done!")
    index.storage_context.persist(persist_dir=data_path)
    print("Persisting done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path-to-db",
        type=str,
        default="db/",
        help="path to chromadb",
    )
    parser.add_argument(
        "--emb-model-path",
        type=str,
        default=None,
        help="path to locally saved sentence transformer model",
    )

    parser.add_argument(
        "--data-path", type=str, help="Path to json files with unstructured chunks"
    )
    parser.add_argument("--output", help="output directory")
    args = parser.parse_args()
    settings = chromadb.get_settings()
    settings.allow_reset = True
    print(f"creating/loading db at {args.path_to_db}...")
    db = chromadb.PersistentClient(path=args.path_to_db, settings=settings)
    print("Done!")
    if args.emb_model_path.startswith("http"):
        print(f"Using Embedding API model endpoint: {args.emb_model_path}")
        embed_model = OpenAIEmbedding(api_base=args.emb_model_path, api_key="dummy", embedding_ctx_length=512, chunk_size=32, tiktoken_enabled=False )
    else:
        print("Loading {}...".format(args.emb_model_path))
        embed_model = HuggingFaceEmbedding(args.emb_model_path)
    main(args.data_path, embed_model, db)
    if args.output:
        shutil.copytree(args.path_to_db)

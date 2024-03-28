import chromadb

import argparse
import os
import json
import shutil
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.storage import StorageContext
import torch


def main(data_path, embed_model, db):
    print("Done!")
    collection = db.get_or_create_collection(
        name="documents", metadata={"hnsw:space": "cosine"}
    )
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    docs = []
    index = VectorStoreIndex(docs, storage_context=storage_context, embed_model=embed_model)
    for dirpath, dirs, files in os.walk(data_path):
        for file in files:
            input_file = os.path.join(dirpath, file)

            with open(input_file, "r") as f:
                input_text = json.load(f)
                for doc in input_text:
                    if doc["type"] == "Table":
                        text = doc["metadata"]["text_as_html"]
                    else:
                        text = doc["text"]
                    if "url" in doc["metadata"]:
                        source = doc["metadata"]["url"]
                    elif "filename" in doc["metadata"]:
                        source = doc["metadata"]["filename"]
                    else:
                        source = "Unknown"
                    metadata = {
                            "Source": source,
                            "Page Number": doc["metadata"]["page_number"],
                            "Commit": os.environ.get("PACH_JOB_ID","")} 
                    docs.append(TextNode(text=text, metadata=metadata))
    print("Number of chunks: ", len(docs))

    for node in docs:
        index.insert_nodes(node)
    index.storage_context.persist(persist_dir=data_path)


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
    settings.allow_reset = False
    print(f"creating/loading db at {args.path_to_db}...")
    db = chromadb.PersistentClient(path=args.path_to_db, settings=settings)
    print("Done!")
    print("Loading {}...".format(args.emb_model_path))
    embed_model = HuggingFaceEmbedding(args.emb_model_path)
    main(args.data_path, embed_model, db)
    if args.output:
        shutil.copytree(args.path_to_db)

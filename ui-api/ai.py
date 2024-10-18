import asyncio
import json
import os
import time

import chromadb
from fastapi.encoders import jsonable_encoder
from iso639 import languages
from llama_index.core import (Settings, VectorStoreIndex,
                              get_response_synthesizer)
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores.types import (MetadataFilter,
                                                  MetadataFilters)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.chroma import ChromaVectorStore
from loguru import logger
from openai import OpenAI

from models import (ConfigItem, DefaultConfig, Event, GeneratePostResponse,
                    ReferenceData, ValueType)


class AIPipeline:
    def __init__(
        self,
        chat_model,
        chat_model_url,
        embed_model,
        similarity_cutoff=0.6,
        top_k=5,
        temp=0.2,
        max_tokens=600,
        system_prompt="",
        streaming=False,
        db_path="./db",
    ):
        self.chat_model_url = chat_model_url
        self.chat_model = (
            chat_model if chat_model else list(self.get_models().keys())[0]
        )
        self.embed_model = embed_model
        self.similarity_cutoff = similarity_cutoff
        self.top_k = top_k
        self.temp = temp
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.db_path = db_path
        self.streaming = streaming

    def get_config(self):
        config = DefaultConfig(
            [
                ConfigItem(
                    name="modelTemperature",
                    friendlyName="Model Temperature",
                    minValue=0.0,
                    maxValue=1.0,
                    valueType=ValueType.FLOAT,
                    defaultValue=self.temp,
                    description="The randomness of the LLM response. Higher is more random, lower is more deterministic",
                ),
                ConfigItem(
                    name="topK",
                    friendlyName="Top K",
                    minValue=0,
                    maxValue=25,
                    valueType=ValueType.INT,
                    defaultValue=self.top_k,
                    description="The Maximum number of chunks to retreive from the vector DB ",
                ),
                ConfigItem(
                    name="maxOutputTokens",
                    friendlyName="Maximum Output Tokens",
                    minValue=0,
                    maxValue=5000,
                    valueType=ValueType.INT,
                    defaultValue=self.max_tokens,
                    description="The Maximum number of tokens for the LLM to generate ",
                ),
                ConfigItem(
                    name="systemPrompt",
                    friendlyName="LLM Instructions",
                    minValue=0,
                    maxValue=5000,
                    valueType=ValueType.STRING,
                    defaultValue=self.system_prompt,
                    description="The Instructions for the model ",
                ),
                ConfigItem(
                    name="similarityCutoff",
                    friendlyName="Similarity Cutoff (%)",
                    minValue=0,
                    maxValue=100,
                    valueType=ValueType.INT,
                    defaultValue=(self.similarity_cutoff * 100),
                    description="Minimum Similarity Score for retrieved chunks",
                ),
            ]
        )
        return config

    def get_models(self):
        model_urls = str(self.chat_model_url).split(",")
        available_models = {}
        if len(model_urls) >= 1:
            for url in model_urls:
                client = OpenAI(base_url=url, api_key="fake")
                models = client.models.list().data
                for model in models:
                    available_models[model.id] = url
        else:
            client = OpenAI(base_url=model_urls[0], api_key="fake")
            models = client.models.list().data
            for model in models:
                available_models[model.id] = model_urls[0]
        return available_models

    def load_llm(self, url=None):
        if url is None:
            modelpath = str(self.chat_model_url.split(",")[0])
        else:
            modelpath = url
        logger.info(f"Using OpenAPI-compatible LLM endpoint: {modelpath}")

        llm = OpenAILike(model="", api_base=modelpath, api_key="fake")
        return llm

    def create_filters(self, tags: list):
        meta_filters = []
        for tag in tags:
            meta_filters.append(MetadataFilter(key="Tag", value=tag))
        filters = MetadataFilters(
            filters=meta_filters,
            condition="or",
        )
        return filters

    def create_query_engine(self, index, filters=None, cutoff=None, top_k=None):
        if cutoff is None:
            cutoff = self.similarity_cutoff
        if top_k is None:
            top_k = self.top_k
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
        return query_engine

    def load_data(self):
        logger.info(f"Using OpenAPI-compatible Embedding endpoint: {self.embed_model}")
        embed_model = OpenAIEmbedding(api_base=self.embed_model, api_key="dummy")
        chroma_client = chromadb.PersistentClient(self.db_path)
        chroma_collection = chroma_client.get_collection(name="documents")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model,
        )
        return index, chroma_collection.get()

    def source_list(self, chunks):
        uploaded_files = {}
        for i in range(len(chunks["ids"])):
            file = chunks["metadatas"][i]["Source"]
            eltags = chunks["metadatas"][i]["Tag"]
            if eltags not in uploaded_files:
                uploaded_files[eltags] = []
            if file not in uploaded_files[eltags]:
                uploaded_files[eltags].append(file)
        return uploaded_files

    def get_filters(self, filter_tags=[]):
        meta_filters = []
        for tag in filter_tags:
            meta_filters.append(MetadataFilter(key="Tag", value=tag))
        return MetadataFilters(
            filters=meta_filters,
            condition="or",
        )

    def output_stream(self, llm_stream, output_nodes):
        references = self.format_references(output_nodes)
        resp = GeneratePostResponse(event=Event.reference, data=references)
        yield f"{json.dumps(jsonable_encoder(resp))}\n"
        for chunk in llm_stream:
            stuff = GeneratePostResponse(event=Event.answer, data=chunk.delta)
            yield f"{json.dumps(jsonable_encoder(stuff))}\n"

    def generate_response(
        self, query, system_prompt, model, model_args, query_engine, streaming=True
    ):
        generate_kwargs = {
            "temperature": (
                model_args["model_temperature"]
                if model_args["model_temperature"] is not None
                else self.temp
            ),
            "top_p": 0.5,
            "max_tokens": (
                model_args["max_output_tokens"]
                if model_args["max_output_tokens"] is not None
                else self.max_tokens
            ),
        }
        logger.info(f"Submitted Args: {model_args}")
        logger.info(f"Generation Args: {generate_kwargs}")
        if model:
            models = self.get_models()
            llm = self.load_llm(models[model])
            llm.model = model
        else:
            llm = self.load_llm()
            llm.model = self.chat_model
        logger.info(f"Querying with: {query}")
        output = query_engine.query(query)
        context_str = ""
        if not system_prompt:
            system_prompt = self.system_prompt
        logger.info(f"system prompt: {system_prompt}")
        for node in output.source_nodes:
            logger.info(f"Context: {node.metadata}")
            context_str += node.text.replace("\n", "  \n")
        text_qa_template_str_llama3 = f"""
            <|begin_of_text|><|start_header_id|>user<|end_header_id|>
            Context information is
            below.
            ---------------------
            {context_str}
            ---------------------
            Using
            the context information, answer the question: {query}
            {system_prompt}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
        if self.streaming:
            logger.info(f"Requesting model {llm.model} for streaming")
            return (
                llm.stream_complete(
                    text_qa_template_str_llama3, formatted=True, **generate_kwargs
                ),
                output.source_nodes,
            )
        else:
            logger.info(f"Requesting model {llm.model} for completion")
            output_response = llm.complete(text_qa_template_str_llama3)
            return output_response, output.source_nodes

    def format_references(self, source_nodes):
        refs_to_return = []
        project = os.getenv("PPS_PROJECT_NAME", "default")
        doc_repo = os.getenv("DOCUMENT_REPO", "documents")
        proxy_url = os.getenv("PACH_PROXY_EXTERNAL_URL_BASE", "http://localhost:30080")
        references = source_nodes
        for i in range(len(references)):
            title = references[i].node.metadata["Source"]
            page = references[i].node.metadata["Page Number"]
            text = references[i].node.text if "text_as_html" not in references[i].node.metadata else references[i].node.metadata["text_as_html"]
            score = round((references[i].score * 100), 3)
            commit = references[i].node.metadata["Commit"]
            doctag = references[i].node.metadata["Tag"]
            if doctag:
                doctag = doctag.replace(" ", "%20")
                out_link = f"{proxy_url}/proxyForward/pfs/{project}/{doc_repo}/{commit}/{doctag}/{title}#page={page}"
            else:
                out_link = f"{proxy_url}/proxyForward/pfs/{project}/{doc_repo}/{commit}/{title}#page={page}"
            if title.startswith("http"):
                out_link = title
            meta = {}
            if "original" in references[i].node.metadata:
                original = references[i].node.metadata["original"]
                if "lang" in references[i].node.metadata:
                    lang = languages.get(
                        alpha2=references[i].node.metadata["lang"]
                    ).name
                    if lang == "Croatian":
                        lang = "Serbian"
                else:
                    lang = "Unknown"
                meta["Original Language"] = lang
                meta["Original Text"] = original
            ref = ReferenceData(
                source=title,
                text=text,
                similarityScore=score,
                page=page,
                url=out_link,
                metadata=meta,
            )
            refs_to_return.append(ref)
        return refs_to_return

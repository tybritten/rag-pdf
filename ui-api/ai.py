import os
import json
from loguru import logger
import chromadb
from llama_index.core import Settings, VectorStoreIndex, get_response_synthesizer
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.chroma import ChromaVectorStore
from models import DefaultConfig, ReferenceData, GeneratePostResponse, Event
from openai import OpenAI
from fastapi.encoders import jsonable_encoder


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
        self.chat_model = chat_model if chat_model else self.get_models()[0]
        self.embed_model = embed_model
        self.similarity_cutoff = similarity_cutoff
        self.top_k = top_k
        self.temp = temp
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.db_path = db_path
        self.streaming = streaming

    def get_config(self):
        return DefaultConfig(
            modelTemperature=self.temp,
            topK=self.top_k,
            maxOutputTokens=self.max_tokens,
            systemPrompt=self.system_prompt,
            similarityCutoff=self.similarity_cutoff,
        )

    def get_models(self):
        modelpath = str(self.chat_model_url)
        client = OpenAI(base_url=modelpath, api_key="fake")
        models = client.models.list().data
        available_models = []
        for model in models:
            available_models.append(model.id)
        return available_models

    def load_llm(self):
        print(Settings.llm)
        logger.info(f"Using OpenAPI-compatible LLM endpoint: {self.chat_model_url}")
        modelpath = str(self.chat_model_url)
        llm = OpenAILike(model="", api_base=modelpath, api_key="fake")
        return llm

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
        yield json.dumps(jsonable_encoder(resp))
        for chunk in llm_stream:
            stuff = GeneratePostResponse(event=Event.answer, data=chunk.delta)
            yield json.dumps(jsonable_encoder(stuff))

    def generate_response(
        self, query, system_prompt, model, model_args, query_engine, streaming=True
    ):

        generate_kwargs = {
            "temperature": (
                model_args["model_temperature"]
                if "model_temperature" in model_args
                else self.temp
            ),
            "top_p": 0.5,
            "max_tokens": model_args["max_tokens"] if "max_tokens" in model_args else self.max_tokens,
        }
        llm = self.load_llm()
        logger.info(f"Querying with: {query}")
        output = query_engine.query(query)

        llm.model = model if model else self.chat_model
        context_str = ""
        if not system_prompt:
            system_prompt = self.system_prompt
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
            the context information, answer the question: {query}
            {system_prompt}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
        if self.streaming:
            logger.info(f"Requesting model {llm.model}")
            return (
                llm.stream_complete(
                    text_qa_template_str_llama3, formatted=True, **generate_kwargs
                ),
                output.source_nodes,
            )
        else:
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
            text = references[i].node.text
            score = round((references[i].score * 100), 3)
            commit = references[i].node.metadata["Commit"]
            doctag = references[i].node.metadata["Tag"]
            newtext = text.encode("unicode_escape").decode("unicode_escape")
            if doctag:
                doctag = doctag.replace(" ", "%20")
                out_link = f"[Link to file in Commit {commit}]({proxy_url}/proxyForward/pfs/{project}/{doc_repo}/{commit}/{doctag}/{title}#page={page})\n"
            else:
                out_link = f"[Link to file in Commit {commit}]({proxy_url}/proxyForward/pfs/{project}/{doc_repo}/{commit}/{title}#page={page})\n"
            if title.startswith("http"):
                out_link = title
            ref = ReferenceData(
                source=title, text=text, similarityScore=score, page=page, url=out_link
            )
            refs_to_return.append(ref)
        return refs_to_return

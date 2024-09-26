
/*
title: RAG Pipeline
description: "Creates a RAG pipeline."
args:
- name: input_repo
  description: The name of the input repo.
  type: string
  default: embed-docs
- name: embed_model
  description: The URL to the embedding model (include /v1)
  type: string
  default: "http://embed.mlis.svc.cluster.local/v1"
- name: chat_url
  description: The URL to the chat model (include /v1)
  type: string
  default: "http://llama3.mlis.svc.cluster.local/v1"
- name: chat_model
  description: The model name for chat model (can get from /v1/models)
  type: string
  default: "/mnt/models/Meta-Llama-3.1-8B-Instruct/"
- name: mldm_base_url
  description: 'The base URL of the MLDM instance.'
  type: string
- name: service_type
  description: What type of K8s service (LoadBalancer, NodePort, ClusterIP)
  type: string
  default: "NodePort"
- name: external_port
  description: The port for the K8s Service
  type: string
  default: "32080"
*/


function(input_repo="embed-docs", embed_model="http://embed.mlis.svc.cluster.local/v1", chat_url="http://llama3.mlis.svc.cluster.local/v1", chat_model="", mldm_base_url="http://localhost:30080", service_type="NodePort", external_port="32080")
{
  "pipeline": {
    "name": "api",
  },
    "transform": {
        "cmd": [
            "./startup.sh"
        ],
        "env": {
            "OPENAI_API_KEY": "fake",
            "DOCUMENT_REPO": input_repo,
            "PACH_PROXY_EXTERNAL_URL_BASE": mldm_base_url,
            "PYTHON_UNBUFFERED": "1",
            "EMBED_MODEL": embed_model,
            "CHAT_MODEL_BASE_URL": chat_url,
            "DB_PATH": "/pfs/data",
            "MAX_TOKENS": "2048",
            "DEFAULT_CHAT_MODEL": chat_model
        },
        "image": "vmtyler/pdk:ui-api-0.0.3"
    },
  "input": {
    "pfs": {
      "repo": input_repo,
      "name": "data",
      "glob": "/"
    }
  },
  "autoscaling": false,
  "service": {
    "type": service_type,
    "externalPort": std.parseInt(external_port),
    "internalPort": 5000
  },
}


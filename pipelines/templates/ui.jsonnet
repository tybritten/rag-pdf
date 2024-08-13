
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
- name: chat_model
  description: The URL to the chat model (do NOT include /v1)
  type: string
  default: "http://llama3.mlis.svc.cluster.local/v1"
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

local chat_cmd(chat_model, embed_model) =
    "streamlit run gui.py -- --path-to-db /pfs/data --path-to-chat-model " + chat_model + " --emb-model-path " + embed_model + " --cutoff 0.6";

function(input_repo="embed-docs", embed_model="http://embed.mlis.svc.cluster.local/v1", chat_model="http://llama3.mlis.svc.cluster.local/v1", mldm_base_url, service_type="NodePort", external_port="32080")
{
  "pipeline": {
    "name": "gui",
  },
  "transform": {
    "image": "vmtyler/pdk:gui-v0.1.6c",
    "cmd": [
      "/bin/bash",
      "-C"
    ],
    "stdin": [chat_cmd(chat_model, embed_model)],
    "env": {
      "PYTHON_UNBUFFERED": "1",
      "PACH_PROXY_EXTERNAL_URL_BASE": mldm_base_url,
      "DOCUMENT_REPO": input_repo
    }
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
    "internalPort": 8501
  },
}


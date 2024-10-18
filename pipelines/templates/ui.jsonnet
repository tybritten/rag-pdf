
/*
title: RAG Pipeline
description: "Creates a RAG pipeline."
args:
- name: input_repo
  description: The name of the input repo.
  type: string
  default: embed-docs
- name: api_url
  description: The URL to the API
  type: string
  default: "http://document-rag-api-v26-user"
- name: app_title
  description: The title of the app
  type: string
  default: "Retrieval Augmented Generation Demo"
- name: chat_title
  description: The title of the Chat Interface
  type: string
  default: "Chat with your Data"
- name: service_type
  description: What type of K8s service (LoadBalancer, NodePort, ClusterIP)
  type: string
  default: "NodePort"
- name: external_port
  description: The port for the K8s Service
  type: string
  default: "32080"
*/


function(chat_title="Chat with your Data", app_title="Retrieval Augmented Generation Demo",input_repo="embed-docs", api_url="http://document-rag-api-v26-user",  service_type="NodePort", external_port="32080")
{
  "pipeline": {
    "name": "gui",
  },
  "transform": {
    "image": "vmtyler/pdk:rag-ui-0.0.5",
    "cmd": [
      "npm",
      "run",
      "prod"
    ],
    "env": {
      "PYTHON_UNBUFFERED": "1",
      "VITE_SERVER_ADDRESS": api_url,
      "VITE_APP_TITLE": app_title,
      "VITE_CHAT_TITLE": chat_title,
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


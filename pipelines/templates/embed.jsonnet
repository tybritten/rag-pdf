
/*
title: Embed Step for RAG Pipeline
description: "Creates the Embedding Step for a  RAG pipeline."
args:
- name: input_repo
  description: The name of the input repo.
  type: string
  default: parse-docs
- name: embed_model
  description: The URL to the embedding model (include /v1)
  type: string
  default: "http://embed.mlis.svc.cluster.local/v1"
*/

local embed_cmd(embed_model) = 
    "python3 embed.py --data-path /pfs/data --emb-model-path " + embed_model + " --path-to-db /pfs/out/";

function(input_repo="parse-docs", embed_model="http://embed.mlis.svc.cluster.local/v1")

{
  "pipeline": {
    "name": "embed-docs",
  },
  "transform": {
    "image": "vmtyler/pdk:embed-v0.1b",
    "cmd": [
      "/bin/bash",
      "-C"
    ],
    "stdin": [embed_cmd(embed_model)],
    "env": {
      "PYTHON_UNBUFFERED": "1"
    }
  },
  "input": {
    "pfs": {
      "repo": input_repo,
      "name": "data",
      "glob": "/"
    }
  },
  "autoscaling": true
}

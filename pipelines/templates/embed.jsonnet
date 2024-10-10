
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
- name: translate_url
  description: The URL to the optional language translation model (include /v1)
  type: string
  default: "http://llama3.mlis.rag.psdc.lan/v1"
- name: translate_model
  description: The model name for optional translation (can get from /v1/models)
  type: string
  default: "/mnt/models/Meta-Llama-3.1-8B-Instruct/"
*/
local join(a) =
    local notNull(i) = i != null;
    local maybeFlatten(acc, i) = if std.type(i) == "array" then acc + i else acc + [i];
    std.foldl(maybeFlatten, std.filter(notNull, a), []);


local args(translate_url, translate_model, embed_model) = 
    join([
    if translate_url != "" then ["--model-url", translate_url, "--translate-model", translate_model],
    ["--emb-model-path", embed_model, "--path-to-db", "/pfs/out"],
    ]);


function(input_repo="parse-docs", embed_model="http://embed.mlis.svc.cluster.local/v1", translate_model, translate_url)

{
  "pipeline": {
    "name": "embed-docs",
  },
  "transform": {
    "image": "vmtyler/pdk:embed-v0.2.5a",
    "cmd": [
      "/bin/bash",
      "-C"
    ],
    "stdin": ["python3", "embed.py", " --data-path", "/pfs/data"] + args(translate_url, translate_model, embed_model),
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

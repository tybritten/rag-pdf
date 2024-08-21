
/*
title: Parse Step for RAG Pipeline
description: "Creates a Parsing Step for a RAG pipeline."
args:
- name: input_repo
  description: The name of the input repo.
  type: string
  default: documents
*/



function(input_repo)

{
  "pipeline": {
    "name": "parse-docs",
  },
  "transform": {
    "image": "vmtyler/pdk:parsing-v0.1.1",
    "cmd": [
      "/bin/bash",
      "-C"
    ],
    "stdin": [
      "python3 parsing.py --input /pfs/documents --output /pfs/out --chunking_strategy by_title --folder_tags"
    ],
    "env": {
      "PYTHON_UNBUFFERED": "1"
    }
  },
  "input": {
    "pfs": {
      "repo": input_repo,
      "glob": "/*/*"
    }
  },
  "resourceRequests": {
    "cpu": 4,
    "memory": "16Gi",
    "disk": "10Gi"
  },
  "autoscaling": true,
  "parallelismSpec": {
    "constant": 2
  },
}


/*
title: Parse Step for RAG Pipeline
description: "Creates a Parsing Step for a RAG pipeline."
args:
- name: input_repo
  description: The name of the input repo.
  type: string
  default: documents
- name: langs
  description: OCR languages for Tesseract, colon separated for Tesseract (https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html)
  type: string
  default: eng
*/

local parse_cmd(langs='eng') = 
          "python3 parsing.py --input /pfs/documents --output /pfs/out --chunking_strategy by_title --folder_tags  --languages " + langs;

function(input_repo='documents', langs="eng")

{
  "autoscaling": false,
  "input": {
    "pfs": {
      "glob": "/*/*",
      "repo": input_repo
    }
  },
  "parallelismSpec": {
    "constant": 2
  },
  "pipeline": {
    "name": "parse-docs"
  },
  "resourceRequests": {
    "cpu": 4,
    "disk": "10Gi",
    "memory": "16Gi"
  },
  "transform": {
    "cmd": [
      "/bin/bash",
      "-C"
    ],
    "env": {
      "PYTHON_UNBUFFERED": "1"
    },
    "image": "vmtyler/pdk:parsing-v0.2.0c",
    "stdin": [parse_cmd(langs)]
  }
}


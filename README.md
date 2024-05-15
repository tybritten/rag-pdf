img src="https://raw.githubusercontent.com/hpe-design/logos/master/Requirements/color-logo.png" alt="HPE Logo" height="100"/>

# RFP RAG demo (Chat with HPE Press Release version)

<b>Author:</b> Tyler Britten, Andrew Mendez </br>
<b>Date:</b> 05/01/2024</br>
<b>Revision:</b> 0.1</br>

This demonstration was built to showcase Retrieval Augmented Generation (RAG) on internal word documents. These word documents contain Request for Proposal (RFP). A request for proposal (RFP) is a business document that announces a project, describes it, and solicits bids from qualified contractors to complete it. It shows how RAG can be used to assist business development teams to ask questions from previous RFPs written.

To replicate this demo, you will need:

 - A functioning Kubernetes cluster with load balancers for external facing services configured
     - cluster having shared mounted folder `/nvmefs1/tyler.britten`
 - Pachyderm/HPE MLDM 2.9.2 installed on the cluster and fully functional
 - At least 1x NVIDIA T4 80GB GPUs
 - Determined.AI/HPE MLDE environment for finetuning models (not included in the base code here)
 - downloaded word documents (in .docx format) that 

<b>NOTE:</b> You might be able to replicate this demo with other GPUs (for example L40s) as well, but you need to consider the memory footprint of other GPUs and adjust accordingly.


## Recorded demos of RAG demo

[ToDo]

## Implementation Overview


- Step 1: Connect to deployed MLDM application
- Step 2: Create MLDM project named `rfp-demo-hpe`
- Step 3: Set new project as current context
- Step 4: Create repo `documents` to hold xml documents
- Step 5: Upload xml documents
- Step 6: Pipeline step to parse documents
- Step 7: Pipeline step to chunk documents
- Step 8: Pipeline step to embed documents using embedding model `bge-large-en-v1.5`
- Step 9: Pipeline step to deploy GUI application
- Step 10: Interact with GUI application
- Step 11: Add new documents to repo `documents` to improve RAG App
- Step 12: (Optional Step) Pipeline step to develop dataset for finetuning embeddings: qna pipeline
- Step 13: (Optional Step) Pipeline step to finetune embeddings
- Step 14: Delete pipelines


<b>RAG demo includes solution components from:</b>
- [Pachyderm / HPE MLDM](https://www.hpe.com/us/en/hpe-machine-learning-data-management-software.html)
- [ChromaDB](https://www.trychroma.com/)
- [Streamlit frontend](https://streamlit.io)

# Steps to run the demo

## Prep Step:

We wont be providing word documents to protect the intellectual property of HPE's RFPs. You can download word documents of internal HPE documents here [link](link). When you are done downloading, make sure to place them in a folder called `data/`

## Step 1: Connect to deployed MLDM application

`pachctl connect pachd-peer.pachyderm.svc.cluster.local:30653`

## Step 2: Create MLDM project named `rfp-demo-hpe`

`pachctl create project rfp-demo-hpe`

## Step 3: Set new project as current context

`pachctl config update context --project rfp-demo-hpe`

### Pipeline will be available at the url:

`http://mldm-pachyderm.us.rdlabs.hpecorp.net/lineage/rfp-demo-hpe`

## Step 4: Create repo `documents` to hold xml documents

`pachctl create repo documents`

## Step 5: Upload single xml document

`pachctl put file -r documents@master -f data/`

## Step 6: Pipeline step to parse documents

This pipeline takes the raw xml documents and parses them into json format.

`pachctl create pipeline -f pipelines/parsing.pipeline.json`

## Step 7: Pipeline step to chunk documents

This pipeline takes the parsed documents and applies chunking to create chunked documents. 

`pachctl create pipeline -f pipelines/chunking.pipeline.json`

## Step 8: Pipeline step to embed documents using embedding model `bge-large-en-v1.5`

This pipeline takes the chunked documents and creates vector embeddings using the vector embedding `bge-large-en-v1.5` 

`pachctl create pipeline -f pipelines/embedding.pipeline.json`


## Step 9: Pipeline step to deploy GUI application

This pipeline will deploy a streamlit application for user to interact with the GUI

`pachctl create pipeline -f pipelines/gui.pipeline.json`

## Step 10: Interact with GUI application

Note: There is an issue with the Houston cluster where there are not enough IP addresses for service pipeline. 

### run command to ssh into node:

`ssh andrew@mlds-mgmt.us.rdlabs.hpecorp.net -L 8080:localhost:8080`

### Find IP of deployed GUI

look at pachyderm GUI pipeline UI element, and you should see an IP
```
Service IP
10.182.1.153:80
```
### Open web browser and go to url `10.182.1.153:80` 

####  Ask in the UI:

`What is HPE's solution offering for AI?`


# Optional Steps

## Step 12: (Optional Step) Pipeline step to develop dataset for finetuning embeddings: qna pipeline 

`pachctl create pipeline -f pipelines/qna.pipeline.json`

## Step 13: (Optional Step) Pipeline step to finetune embeddings 

Note, what is hardcoded is the following in `finetune/experiment/const.yaml`:

Make sure you create a workspace named `Tyler` and project `doc_embeds` to use the same `finetune/experiment/const.yaml`
```
name: arctic-embed-fine-tune
workspace: Tyler
project: doc_embeds
```

Also, the bind_mounts are hardcoded. This assumes you are running on a cluster (i.e. the houston cluster) where you have a mounted shared folder called `/nvmefs1` 

```
bind_mounts:
  - container_path: /nvmefs1/
    host_path: /nvmefs1/
    propagation: rprivate
    read_only: false
  - container_path: /determined_shared_fs
    host_path: /nvmefs1/determined/checkpoints
    propagation: rprivate
    read_only: false
```

### Step to trigger finetune pipeline:  

`pachctl create pipeline -f pipelines/finetune.pipeline.json`


## Step 14: Delete pipelines

`pachctl delete pipeline gui`
`pachctl delete pipeline finetune-embedding`
`pachctl delete pipeline generate-qna`
`pachctl delete pipeline embed-docs`
`pachctl delete pipeline chunk-doc`
`pachctl delete pipeline parse-docs`
`pachctl delete repo documents`

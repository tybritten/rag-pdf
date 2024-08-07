import nltk

nltk.download("stopwords", download_dir="/nvmefs1/andrew.mendez/nltk_cache")

import logging
import os

import determined as det
import torch
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from sentence_transformers import (InputExample, SentenceTransformer,
                                   evaluation, losses)
from torch.utils.data import DataLoader

from data import download_pach_repo


def download_data(data_config, data_dir):

    files = download_pach_repo(
        data_config["pachyderm"]["host"],
        data_config["pachyderm"]["port"],
        data_config["pachyderm"]["repo"],
        data_config["pachyderm"]["branch"],
        data_dir,
        data_config["pachyderm"]["token"],
        data_config["pachyderm"]["project"],
        data_config["pachyderm"]["previous_commit"],
    )
    print(f"Data dir set to : {data_dir}")
    return [des for src, des in files]


def main(core_context):
    info = det.get_cluster_info()
    data_config = info.user_data
    hparams = info.trial.hparams
    data_dir = "/tmp/data"
    if data_config["pachyderm"]["host"] is not None:
        os.makedirs(data_dir, exist_ok=True)
        download_data(data_config, data_dir)
    ckpt_dir = "/tmp/checkpoint"

    dataset = EmbeddingQAFinetuneDataset.from_json(f"{data_dir}/train_dataset.json")
    training_data = []
    for query_id, query in dataset.queries.items():
        for node_id in dataset.relevant_docs[query_id]:
            text = dataset.corpus[node_id]
            example = InputExample(texts=[query, text], label=0.0)
            training_data.append(example)

    dataloader = DataLoader(
        training_data,
        batch_size=hparams.get("global_batch_size"),
        shuffle=True,
    )

    val_dataset = EmbeddingQAFinetuneDataset.from_json(f"{data_dir}/test_dataset.json")

    model = SentenceTransformer(hparams.get("model_id"))

    train_loss = losses.CosineSimilarityLoss(model)

    max_steps = len(dataloader) * hparams["epochs"]

    evaluator = evaluation.InformationRetrievalEvaluator(
        val_dataset.queries, val_dataset.corpus, val_dataset.relevant_docs
    )
    op = next(iter(core_context.searcher.operations()))

    def training_callback(score, epoch, steps):
        if steps == -1:
            # This is the end of the epoch
            steps = len(dataloader)
            batches = steps + epoch * len(dataloader)
            op.report_progress(batches / max_steps)
            storage_id = core_context.checkpoint.upload(
                ckpt_dir, metadata={"steps_completed": batches}, shard=False
            )
            logging.info(f"done uploading checkpoint {storage_id}")
            core_context.train.report_validation_metrics(
                steps_completed=batches, metrics={"eval_loss": score}
            )
            if core_context.preempt.should_preempt():
                exit(0)
        else:
            batches = steps + epoch * len(dataloader)
            op.report_progress(batches / max_steps)
            core_context.train.report_training_metrics(
                steps_completed=batches, metrics={"loss": score}
            )
        logging.info(
            f"Epoch {epoch} - Step {steps} - Batches: {batches} - Loss: {score}"
        )

    warmup_steps = int(len(dataloader) * hparams["epochs"] * 0.1)

    model.fit(
        train_objectives=[(dataloader, train_loss)],
        epochs=hparams["epochs"],
        warmup_steps=warmup_steps,
        output_path=ckpt_dir,
        evaluation_steps=hparams["eval_steps"],
        callback=training_callback,
        evaluator=evaluator,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    with det.core.init() as core_context:
        main(core_context)

import argparse
import json
import os
from pathlib import Path

import torch
from loguru import logger
from rag_schema import DataElement, DataType, Document, Metadata
from unstructured.partition.auto import partition
from langdetect import detect, LangDetectException

parser = argparse.ArgumentParser(description="File Parser")
parser.add_argument("--input", type=str, help="input directory")
parser.add_argument("--output", default="./output", help="output directory")
parser.add_argument("--strategy", default="auto", help="parsing strategy")
parser.add_argument("--chunking_strategy", default=None, help="chunking strategy")
parser.add_argument(
    "--folder_tags",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="folder tags",
)
parser.add_argument("--languages", default="eng", type=str, help="ocr languages separated by : eng:rus etc from https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html")


def elements_to_rag_schema(elements: list, languages, tag=None):
    output_list = Document()
    for element in elements:
        el = element.to_dict()
        if "url" in el["metadata"]:
            source = el["metadata"]["url"]
        elif "filename" in el["metadata"]:
            source = el["metadata"]["filename"]
        else:
            source = "Unknown"
        el["metadata"]["source"] = source
        page_number = el["metadata"].get("page_number", 1)
        url = el["metadata"].get("url", None)
        text_as_html = el["metadata"].get("text_as_html", None)
        if (len(languages) > 1 or "eng" not in languages) and len(el["text"]) > 30:
            try:
                lang = detect(el["text"])
            except LangDetectException:
                logger.error(f'Lang detect for text {el["text"]} failed')
                lang = ''
        else:
            lang = ''
        output_list.append(
            DataElement(
                id=el["element_id"],
                data_type=DataType(el["type"]),
                content=el["text"],
                metadata=Metadata(
                    source=el["metadata"]["source"],
                    page_number=page_number,
                    url=url,
                    text_as_html=text_as_html,
                    lang=lang,
                    tag=tag,
                ),
            )
        )
    return output_list



def parse(input_file, output, strategy, chunking_strategy, languages, tag=None):
    logger.info(f"Processing {input_file}")
    elements = partition(
        filename=input_file,
        skip_infer_table_types=[],
        strategy=strategy,
        chunking_strategy=chunking_strategy,
        languages=languages
    )
    output_list = elements_to_rag_schema(elements, languages, tag=tag)
    output_path = os.path.join(output, Path(input_file).stem + ".json")
    with open(output_path, "w") as f:
        logger.info(f"Writing output to {output_path}")
        json.dump(output_list, f, indent=4)


def parse_url(url, output, strategy, chunking_strategy, languages, tag=None):
    logger.info(f"Processing {url}")
    elements = partition(
        url=url,
        skip_infer_table_types=[],
        strategy=strategy,
        chunking_strategy=chunking_strategy,
        languages=languages
    )
    output_list = elements_to_rag_schema(elements, languages, tag=tag)
    output_path = os.path.join(output, Path(url).stem + ".json")
    with open(output_path, "w") as f:
        logger.info(f"Writing output to {output_path}")
        json.dump(output_list, f, indent=4)


def main(args):
    languages = args.languages.split(':')
    for dirpath, dirs, files in os.walk(args.input):
        for file in files:
            input_file = os.path.join(dirpath, file)
            if file.startswith("."):
                print(input_file)
            elif input_file.endswith(".url"):
                logger.info(f"Processing URL file: {input_file}")
                with open(input_file) as file:
                    lines = [line.rstrip() for line in file]
                for url in lines:
                    logger.info(f"Processing {url}")
                    if args.folder_tags:
                        tag = dirpath.replace(args.input, "")
                        if tag.endswith("/"):
                            tag = tag[:-1]
                        if tag.startswith("/"):
                            tag = tag[1:]
                    else:
                        tag = None
                    parse_url(
                        url, args.output, args.strategy, args.chunking_strategy, languages, tag
                    )
            else:
                if args.folder_tags:
                    tag = dirpath.replace(args.input, "")
                    if tag.endswith("/"):
                        tag = tag[:-1]
                    if tag.startswith("/"):
                        tag = tag[1:]
                    if "/" in tag:
                        tag = tag.split("/")[0]
                else:
                    tag = None
                parse(
                    input_file, args.output, args.strategy, args.chunking_strategy, languages, tag
                )


def init():
    logger.info(f"GPU Available: {torch.cuda.is_available()}")


if __name__ == "__main__":
    args = parser.parse_args()
    init()
    logger.info("Starting processing")
    if args.folder_tags:
        logger.info("Using folder names as tags")
    main(args)


import os
import argparse
import json
from pathlib import Path
from loguru import logger
from semantic_router.encoders import HuggingFaceEncoder
from unstructured.chunking.title import chunk_by_title
from unstructured.staging.base import convert_to_dict, dict_to_elements
from rolling_window import UnstructuredSemanticSplitter

parser = argparse.ArgumentParser(description="File Parser")
parser.add_argument("--input", type=str, help="input directory")
parser.add_argument("--output", default="./output", help="output directory")
parser.add_argument("--chunker", default="unstructered", help="chunking engine")
parser.add_argument("--combine_text_under_n_chars", default=50, help="unstructured setting")
parser.add_argument("--max_characters", default=750, help="unstructured setting")
parser.add_argument("--new_after_n_chars", default=500, help="unstructured setting")
parser.add_argument("--embedding_model_path", default="BAAI/bge-base-en-v1.5", help="embedding model for rolling_window")
parser.add_argument("--rolling_min_split", type=int, default=50, help="min split tokens rolling_window")
parser.add_argument("--rolling_max_split", type=int, default=100, help="max split tokens rolling_window")


def chunk_docs_unstruct(args, elements):
    chunking_settings = {
        "combine_text_under_n_chars": args.combine_text_under_n_chars,
        "max_characters": args.max_characters,
        "new_after_n_chars": args.new_after_n_chars,
    }
    chunked_raw = chunk_by_title(elements=elements, **chunking_settings)
    results = convert_to_dict(chunked_raw)
    return results


def chunk_with_rolling_window(args, elements):
    encoder = HuggingFaceEncoder(name=args.embedding_model_path)
    splitter = UnstructuredSemanticSplitter(
        encoder=encoder,
        window_size=1,  # Compares each element with the previous one
        min_split_tokens=args.rolling_min_split,
        max_split_tokens=args.rolling_max_split
    )
    elements_dict = convert_to_dict(elements)
    results = splitter(elements_dict)
    return results


def main(args):
    for dirpath, dirs, files in os.walk(args.input):
        for file in files:
            input_file = os.path.join(dirpath, file)
            logger.info(f"Processing {input_file}.....")
            with open(input_file) as file:
                contents = json.load(file)
                elements_raw = dict_to_elements(contents)
            if args.chunker == "rolling_window":
                logger.info(f"Processing {input_file} with rolling window")
                elements = chunk_with_rolling_window(args, elements_raw)
            else:
                logger.info(f"Processing {input_file} with unstructured")
                elements = chunk_docs_unstruct(args, elements_raw)
            logger.info(f"Finished processing {input_file} with {len(elements)} chunks")
            output_path = os.path.join(args.output, Path(input_file).stem + ".json")
            with open(output_path, "w") as f:
                logger.info(f"Writing output to {output_path}")
                json.dump(elements, f, indent=4)


if __name__ == '__main__':
    args = parser.parse_args()
    logger.info('Starting processing')
    main(args)

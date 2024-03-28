from unstructured.partition.auto import partition
import os
import argparse
import json
from pathlib import Path
from loguru import logger
import torch


parser = argparse.ArgumentParser(description="File Parser")
parser.add_argument("--input", type=str, help="input directory")
parser.add_argument("--output", default="./output", help="output directory")
parser.add_argument("--strategy", default="auto", help="parsing strategy")
parser.add_argument("--chunking_strategy", default=None, help="chunking strategy")

def parse(input_file, output, strategy, chunking_strategy):
    logger.info(f"Processing {input_file}")
    elements = partition(
        filename=input_file,
        skip_infer_table_types=[],
        pdf_infer_table_structure=True,
        strategy=strategy,
        chunking_strategy=chunking_strategy
    )
    output_path = os.path.join(output, Path(input_file).stem + ".json")
    output_list = []
    for element in elements:
        output_list.append(element.to_dict())
    with open(output_path, "w") as f:
        logger.info(f"Writing output to {output_path}")
        json.dump(output_list, f, indent=4)


def parse_url(url, output, strategy, chunking_strategy):
    logger.info(f"Processing {url}")
    elements = partition(
        url=url,
        skip_infer_table_types=[],
        pdf_infer_table_structure=True,
        strategy=strategy,
        chunking_strategy=chunking_strategy
    )
    output_path = os.path.join(output, Path(url).stem + ".json")
    output_list = []
    for element in elements:
        output_list.append(element.to_dict())
    with open(output_path, "w") as f:
        logger.info(f"Writing output to {output_path}")
        json.dump(output_list, f, indent=4)


def main(args):
    for dirpath, dirs, files in os.walk(args.input):
        for file in files:
            input_file = os.path.join(dirpath, file)
            if input_file.endswith(".url"):
                logger.info(f"Processing URL file: {input_file}")
                with open(input_file) as file:
                    lines = [line.rstrip() for line in file]
                for url in lines:
                    logger.info(f"Processing {url}")
                    parse_url(url, args.output, args.strategy, args.chunking_strategy)
            else:
                parse(input_file, args.output, args.strategy, args.chunking_strategy)
            

def init():
    logger.info(f"GPU Available: {torch.cuda.is_available()}")


if __name__ == '__main__':
    args = parser.parse_args()
    init()
    logger.info('Starting processing')
    main(args)

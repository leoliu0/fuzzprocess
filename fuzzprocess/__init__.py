#!/bin/python
import argparse
import csv
import time

import faiss
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("file1", type=str, help="input file1")
parser.add_argument("file2", type=str, help="input file2")
parser.add_argument("-k", type=int, required=False, help="Top k to search", default=10)
parser.add_argument(
    "-d", type=int, required=False, help="minimum distance 0-100", default=20
)
parser.add_argument("-b", action="store_true", help="Use the best model")
parser.add_argument("-m", type=str, help="Pass your model name")
parser.add_argument("--cpu", action="store_true", help="Force use CPU")
parser.add_argument(
    "-o", type=str, required=False, help="output file name", default="__fuzz_result.csv"
)
args = parser.parse_args()

distance = args.d / 100


def read_str(file):
    with open(file) as f:
        return [x.replace('"', "").strip() for x in f.readlines() if x.strip()]


def select_model():
    if args.m:
        name = args.m
    elif args.b:
        name = "dunzhang/stella_en_400M_v5"
    else:
        name = "sentence-transformers/all-MiniLM-L6-v2"
    logger.info(f"Loading Model {name}")

    if args.cpu:
        device = "mps"
    else:
        device = "cuda"
    return SentenceTransformer(name, trust_remote_code=True, device=device)


def main():
    print(f"Searching {args.k} and minimum distance is {distance*100}")
    start = time.time()
    s1 = read_str(args.file1)
    s2 = read_str(args.file2)

    model = select_model()
    logger.info("Encoding s1")
    e1 = model.encode(s1, batch_size=100, show_progress_bar=True)
    logger.info("Encoding s2")
    e2 = model.encode(s2, batch_size=100, show_progress_bar=True)

    index = faiss.IndexFlatIP(model[1].word_embedding_dimension)

    faiss.normalize_L2(e1)
    faiss.normalize_L2(e2)

    index.add(e2)

    logger.info("Start searching")

    with open(args.o, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["s1", "s2", "n", "cos"])

        if len(e1) > 10_0000 and len(e2) > 10_0000:
            e1 = np.array_split(e1, 10000)
        elif len(e1) > 10000:
            e1 = np.array_split(e1, 100)
        else:
            e1 = [e1]

        counter = 0
        for sub_e1 in tqdm(e1):
            D, I = index.search(sub_e1, args.k)
            for i, (a, di) in enumerate(zip(I, D)):
                for n, (j, dj) in enumerate(zip(a, di)):
                    if j >= 0 and dj > distance:
                        writer.writerow(
                            [s1[i + counter], s2[j], n + 1, round(dj * 100, 2)]
                        )
            counter += len(sub_e1)


if __name__ == "__main__":
    main()

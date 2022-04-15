#!/bin/python
import faiss
import argparse
import csv
from loguru import logger
import time

from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser()
parser.add_argument("file1", type=str, help="input file1")
parser.add_argument("file2", type=str, help="input file2")
parser.add_argument("-k", type=int, required=False, help="Top k to search", default=10)
parser.add_argument(
    "-o", type=str, required=False, help="Top k to search", default="__result.csv"
)
args = parser.parse_args()


def read_str(file):
    with open(file) as f:
        return [x.strip() for x in f.readlines() if x.strip()]


def main():
    s1 = read_str(args.file1)
    s2 = read_str(args.file2)

    logger.info("Loading Model")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    logger.info("Encoding s1")
    e1 = model.encode(s1)
    logger.info("Encoding s2")
    e2 = model.encode(s2)

    index = faiss.IndexFlatIP(384)

    faiss.normalize_L2(e1)
    faiss.normalize_L2(e2)

    index.add(e2)

    logger.info("Start searching")
    start = time.time()
    D, I = index.search(e1, args.k)

    end = (time.time() - start) / 60
    logger.info(f"Searching Finished in {end:.2f} minutes")
    with open(args.o, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["s1", "s2", "n", "cos"])
        for i, (a, di) in enumerate(zip(I, D)):
            for n, (j, dj) in enumerate(zip(a, di)):
                if j >= 0:
                    writer.writerow([s1[i], s2[j], n + 1, dj])


if __name__ == "__main__":
    main()
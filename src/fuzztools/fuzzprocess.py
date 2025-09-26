#!/bin/python
import argparse
import csv
from pathlib import Path

import faiss
import torch
import numpy as np
import polars as p
from loguru import logger
from tqdm.auto import tqdm

# Import shared functions from the new utility script
from .utils import (
    read_strings_from_file,
    default_model,
    best_model,
    embed,
    select_model,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file1", type=str, help="input file1")
    parser.add_argument("file2", type=str, help="input file2")
    parser.add_argument(
        "--load_vec", action="store_true", help="Load Vec instead of csv files"
    )
    parser.add_argument(
        "-k", type=int, required=False, help="Top k to search", default=10
    )
    parser.add_argument(
        "-d", type=int, required=False, help="minimum distance 0-100", default=20
    )
    parser.add_argument(
        "--batch", type=int, required=False, help="Batch Size for Model", default=10
    )
    parser.add_argument("-b", action="store_true", help="Use the best model")
    parser.add_argument("-m", type=str, help="Pass your model name")
    parser.add_argument("--hnsw", action="store_true", help="Use HNSW indexing")
    parser.add_argument("--cpu", action="store_true", help="Force use CPU")
    parser.add_argument(
        "-o",
        type=str,
        required=False,
        help="output file name",
        default="__fuzz_result.csv",
    )
    parser.add_argument(
        "--save_vec", action="store_true", help="Save Vectors into a Parquet File"
    )
    args = parser.parse_args()

    distance = args.d / 100

    logger.add("__fuzzprocess.log", rotation="100 MB", retention="10 days")
    logger.info(f"Searching {args.k} and minimum distance is {distance * 100}")

    if not args.load_vec:
        # Read strings using the utility function
        s1 = read_strings_from_file(args.file1)
        if args.file1 == args.file2:
            s2 = s1
        else:
            s2 = read_strings_from_file(args.file2)

        # Determine which model to use based on arguments
        if args.m:
            model_name = args.m
        elif args.b:
            model_name = best_model
        else:
            model_name = default_model
        model = select_model(model_name, args.cpu)

        # Load the selected model using the utility function
        logger.info("Encoding s1")
        e1 = embed(s1, model, args.batch).astype(np.float32)
        faiss.normalize_L2(e1)
        if args.save_vec:
            p.DataFrame({"s": s1, "v": e1}).write_parquet(
                out1 := Path(args.file1).stem + "_" + Path(model_name).stem + ".parquet"
            )
            logger.info(f"vector file {out1} saved")

        logger.info("Encoding s2")
        if args.file1 == args.file2:
            e2 = e1
        else:
            e2 = embed(s2, model, args.batch).astype(np.float32)
        faiss.normalize_L2(e2)

        if args.save_vec:
            p.DataFrame({"s": s2, "v": e2}).write_parquet(
                out2 := Path(args.file2).stem + "_" + Path(model_name).stem + ".parquet"
            )
            logger.info(f"vector file {out2} saved")
    else:  # if load pre-computed vec
        f1 = p.read_parquet(args.file1)
        if args.file1 == args.file2:
            f2 = f1
        else:
            f2 = p.read_parquet(args.file2)
        s1, s2 = f1["s"].to_list(), f2["s"].to_list()
        e1, e2 = np.vstack(f1["v"]), np.vstack(f2["v"])

    if args.hnsw:
        index = faiss.IndexHNSWFlat(e1.shape[1], 64)
        logger.info(f"Using HNSW indexes")
    else:
        index = faiss.IndexFlatIP(e1.shape[1])
    index.add(e2)

    logger.info("Start searching")
    with open(args.o, "w", encoding="utf-8") as f:
        logger.info(f"Writing to {args.o}")
        writer = csv.writer(f, delimiter="\t")
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

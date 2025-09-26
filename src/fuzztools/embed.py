import argparse
from pathlib import Path

import faiss
import polars as p
from loguru import logger
from .utils import (
    read_strings_from_file,
    default_model,
    best_model,
    embed,
)


def main():
    """Main function to encode strings and save them to a Parquet file."""
    parser = argparse.ArgumentParser(
        description="Encode strings from a file and save them as vectors."
    )
    parser.add_argument(
        "input_file", type=str, help="Input file with strings to encode."
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default=default_model,
        help="The name of the SentenceTransformer model to use.",
    )
    parser.add_argument("-b", action="store_true", help="Use the best model")
    parser.add_argument("--cpu", action="store_true", help="Force the use of CPU.")
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="Output Parquet file name. Defaults to [input_file_stem]_[model_stem].parquet",
    )
    args = parser.parse_args()

    logger.add("__encode_process.log", rotation="100 MB", retention="10 days")

    strings = read_strings_from_file(args.input_file)

    model_name = best_model if args.b else args.model_name
    embeddings = embed(strings, model_name, args.cpu)

    faiss.normalize_L2(embeddings)

    if args.output_file:
        output_path = args.output_file
    else:
        input_path = Path(args.input_file)
        model_stem = Path(model_name).stem
        output_path = f"{input_path.stem}_{model_stem}.parquet"

    # Step 6: Save the strings and their vectors
    df = p.DataFrame({"s": strings, "v": list(embeddings)})
    df.write_parquet(output_path)
    logger.info(f"Encoded strings and vectors have been saved to {output_path}")


if __name__ == "__main__":
    main()

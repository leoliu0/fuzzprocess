from loguru import logger
from sentence_transformers import SentenceTransformer
from itertools import batched
import numpy as np
from tqdm import tqdm
import os

default_model = "NovaSearch/stella_en_400M_v5"
best_model = "Qwen/Qwen3-Embedding-4B"


def read_strings_from_file(file_path: str) -> list[str]:
    logger.info(f"Reading strings from {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        # This will remove any leading/trailing whitespace and filter out empty lines
        return [line.strip() for line in f if line.strip()]


def select_model(model_name, use_cpu: bool = False):
    device = "cpu" if use_cpu else None
    logger.info(f"Loading model: {model_name}")
    if "gemini" not in model_name:
        return SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device=device,
            model_kwargs={"torch_dtype": "bfloat16"},
        )
    else:
        return "Gemini"


def embed(strings: list, model, batch_size) -> list:
    """
    load a model to embed strings.
    """
    # Set the device to 'cpu' if the --cpu flag is used, otherwise auto-select (GPU if available)

    return model.encode(strings, batch_size=batch_size, show_progress_bar=True)

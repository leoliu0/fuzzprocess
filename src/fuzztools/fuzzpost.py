#!/usr/bin/python
import pandas as pd
import argparse
import time
import os
from pandarallel import pandarallel
from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-2.5-flash-lite"

DEFAULT_SYSTEM_PROMPT = """Are the two firms the same or related company? 
Yes or No. Return No if you cannot determine and need more information or just highly likely.
Explain your reason based on your knowledge."""

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=10_000),
)


def process_row(row, model, sys):
    prompt_content = f"{row['s1']} and {row['s2']}"

    # Sends a query to the generative model and retries on failure.
    for attempt in range(5):
        try:
            # Generate content using the provided model and configuration
            return client.models.generate_content(
                model=model,
                contents=prompt_content,
                config=types.GenerateContentConfig(
                    max_output_tokens=20,
                    temperature=0,
                    system_instruction=[sys],
                    # thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            ).text
        except Exception as e:
            print(f"An error occurred on attempt {attempt + 1}: {e}")
            # Wait for 2 seconds before retrying
            time.sleep(2)
    return


def main():
    """
    Main function to parse arguments and run the processing pipeline.
    """
    # --- Argument Parsing ---
    # Sets up the command-line interface for the script
    parser = argparse.ArgumentParser(
        description="Process company name pairs in parallel using a generative AI model."
    )
    parser.add_argument(
        "input_file",
        help="Path to the input CSV file. Must contain 's1' and 's2' columns.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        help="Path to save the output CSV file.",
        default="__fuzzpost_results.csv",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=20,
        help="Number of parallel jobs to use.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="The generative model to use",
    )
    parser.add_argument(
        "-p",
        "--prompt-file",
        type=str,
        default=None,
        help="Path to a text file containing the system prompt.",
    )

    parser.add_argument(
        "-k",
        type=int,
        default=10,
        help="K Neighbours",
    )

    parser.add_argument(
        "-d",
        type=int,
        default=50,
        help="Similiarity filter",
    )

    args = parser.parse_args()

    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
    except Exception as e:
        print(f"Error configuring Generative AI client: {e}")
        return

    # Initialize the generative model client

    # --- Pandarallel Initialization ---
    # Initializes the parallel processing library with the specified number of workers
    pandarallel.initialize(progress_bar=True, nb_workers=args.jobs)

    # --- Load Data and Prompt ---
    try:
        print(f"Reading input file: {args.input_file}, K={args.k}, D={args.d}")
        df = pd.read_csv(args.input_file, sep="\t")
        df = df[df.n < args.k]
        df = df[df.cos > args.d]
        # Ensure required columns exist
        if "s1" not in df.columns or "s2" not in df.columns:
            raise ValueError("Input CSV must contain 's1' and 's2' columns.")
    except FileNotFoundError:
        print(f"Error: The file '{args.input_file}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return

    # Load system prompt from file or use default
    if args.prompt_file:
        try:
            with open(args.prompt_file, "r") as f:
                system_prompt = f.read()
            print(f"Loaded system prompt from: {args.prompt_file}")
        except FileNotFoundError:
            print(
                f"Warning: Prompt file '{args.prompt_file}' not found. Using default prompt."
            )
            system_prompt = DEFAULT_SYSTEM_PROMPT
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    print("System Prompt: \n ", system_prompt)

    # --- Processing ---
    print("=== Starting parallel processing of the data...")
    print(f"=== Using model {args.model}")
    # The 'parallel_apply' method distributes the work across the initialized workers.
    # We pass the required arguments for our function using a lambda.
    df["model_response"] = df.parallel_apply(
        lambda row: process_row(row, args.model, system_prompt), axis=1
    )

    df['model_response'] = df['model_response'].str.replace('\n', ' ', regex=False)
    df["yes"] = df["model_response"].str[:10].str.lower().str.contains("yes")

    # --- Save Output ---
    try:
        print(f"=== Saving results to: {args.output_file}")
        df.to_csv(args.output_file, index=False, sep="\t")
    except Exception as e:
        print(f"An error occurred while saving the output file: {e}")


if __name__ == "__main__":
    main()

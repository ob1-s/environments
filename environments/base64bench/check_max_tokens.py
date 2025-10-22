import asyncio
import base64
import json
import os
from argparse import ArgumentParser

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# --- Copy the prompt templates directly from your environment code ---
ENCODE_PROMPT_TEMPLATE = """
Encode the following text to base64. Output ONLY the base64-encoded string with no additional formatting, explanations, or markdown.

Text to encode:
{text}

Base64:
""".strip()

DECODE_PROMPT_TEMPLATE = """
Decode the following base64 string. Output ONLY the decoded text with no additional formatting, explanations, or markdown.

Base64 to decode:
{encoded_text}

Decoded text:
""".strip()
# --- End of copied templates ---


async def get_token_count(client: AsyncOpenAI, model: str, prompt_text: str) -> int:
    """
    Gets the token count for a given text by sending a request to an
    OpenAI-compatible API with a minimal max_tokens value.
    """
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=1,
            timeout=10.0,
        )
        if response.usage:
            return response.usage.prompt_tokens
        return 0
    except Exception as e:
        print(f"An error occurred while tokenizing a prompt: {e}")
        return 0


async def main():
    parser = ArgumentParser(description="Calculate the max input token length for the base64 benchmark.")
    parser.add_argument(
        "file",
        type=str,
        help="Path to the eval.jsonl file.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="The model name to use for the API call (e.g., 'Qwen/Qwen3-4B-Instruct-2507').",
    )
    parser.add_argument(
        "--base-url",
        "-b",
        type=str,
        default="http://localhost:8000/v1",
        help="The base URL of the OpenAI-compatible API server.",
    )
    parser.add_argument(
        "--api-key",
        "-k",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        help="API key for the service. Defaults to OPENAI_API_KEY env var or 'EMPTY'.",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=32,
        help="Number of concurrent requests to the tokenization API.",
    )
    args = parser.parse_args()

    print(f"Analyzing token lengths for '{args.file}' using model '{args.model}' at '{args.base_url}'...")

    try:
        with open(args.file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: The file '{args.file}' was not found.")
        return

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = []
    max_token_len = 0
    max_token_prompt = ""
    max_token_type = ""  # <-- ADDED: Variable to store the data type

    async def check_prompt(prompt_text: str, data_type: str):
        nonlocal max_token_len, max_token_prompt, max_token_type
        async with semaphore:
            count = await get_token_count(client, args.model, prompt_text)
            if count > max_token_len:
                max_token_len = count
                max_token_prompt = prompt_text
                max_token_type = data_type  # <-- ADDED: Store the type

    for line in lines:
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
            original_text = sample["text"]
            data_type = sample["type"]  # <-- ADDED: Get the type from the sample

            # 1. Generate the 'encode' prompt
            encode_prompt = ENCODE_PROMPT_TEMPLATE.format(text=original_text)
            tasks.append(check_prompt(encode_prompt, data_type))

            # 2. Generate the 'decode' prompt
            encoded_text = base64.b64encode(original_text.encode("utf-8")).decode("utf-8")
            decode_prompt = DECODE_PROMPT_TEMPLATE.format(encoded_text=encoded_text)
            tasks.append(check_prompt(decode_prompt, data_type))

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Skipping malformed line: {line.strip()} ({e})")

    # Run all tasks concurrently with a progress bar
    await tqdm_asyncio.gather(*tasks, desc="Tokenizing prompts")

    await client.close()

    print("\n--- Analysis Complete ---")
    print(f"Max token length found: {max_token_len}")
    print(f"Data type of max token prompt: {max_token_type}")  # <-- ADDED: Print the type
    print("\nPrompt that resulted in the max token length:")
    print("-" * 40)
    print(max_token_prompt)
    print("-" * 40)


if __name__ == "__main__":
    # To run this, you'll need openai and tqdm:
    # uv pip install openai tqdm
    asyncio.run(main())

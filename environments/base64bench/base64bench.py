import base64
from pathlib import Path

import verifiers as vf
from datasets import Dataset, concatenate_datasets, load_dataset
from Levenshtein import distance as levenshtein_distance

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


def normalized_levenshtein_similarity(str1: str, str2: str) -> float:
    """
    Calculate normalized Levenshtein similarity between two strings.
    Returns a value between 0 and 1, where 1 means identical strings.
    """
    if not str1 and not str2:
        return 1.0
    if not str1 or not str2:
        return 0.0

    dist = levenshtein_distance(str1, str2)
    max_len = max(len(str1), len(str2))
    return 1.0 - (dist / max_len)


def encode_similarity_reward(completion: vf.Messages, answer: str, parser: vf.Parser, **kwargs) -> float:
    model_output = parser.parse_answer(completion)

    if not model_output:
        return 0.0

    try:
        decoded_bytes = base64.b64decode(model_output, validate=True)
        decoded_text = decoded_bytes.decode("utf-8")
        return normalized_levenshtein_similarity(decoded_text, answer.strip())
    except (ValueError, base64.binascii.Error):
        # Failed to decode, means the output was not valid base64.
        return 0.0


def decode_similarity_reward(completion: vf.Messages, answer: str, parser: vf.Parser, **kwargs) -> float:
    model_output = parser.parse_answer(completion)

    if not model_output:
        return 0.0

    return normalized_levenshtein_similarity(model_output, answer.strip())


class Qwen3Parser(vf.Parser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def parse(self, text: str) -> str:
        if "</think>" not in text:
            return ""  # Cropped off before the </think> tag.
        return text.split("</think>")[1].strip()


def load_environment(
    train_file: str = "train.jsonl",
    eval_file: str = "eval.jsonl",
    seed: int = 3301,
    use_think: bool = False,
    task: str | None = None,
    examples_per_data_type: int | None = None,
    **kwargs,
) -> vf.Environment:
    if task not in [None, "encode", "decode"]:
        raise ValueError("`task` must be 'encode', 'decode', or None.")

    env_dir = Path(__file__).parent
    train_path = str(env_dir / train_file)
    eval_path = str(env_dir / eval_file)

    raw_datasets = load_dataset("json", data_files={"train": train_path, "test": eval_path})

    if examples_per_data_type is not None and examples_per_data_type > 0:
        eval_dataset = raw_datasets["test"]

        data_types = eval_dataset.unique("type")

        sampled_subsets = []
        for data_type in data_types:
            subset = eval_dataset.filter(lambda example: example["type"] == data_type)
            shuffled_subset = subset.shuffle(seed=seed)
            num_to_select = min(len(shuffled_subset), examples_per_data_type)
            sampled_subset = shuffled_subset.select(range(num_to_select))
            sampled_subsets.append(sampled_subset)

        balanced_eval_dataset = concatenate_datasets(sampled_subsets)
        raw_datasets["test"] = balanced_eval_dataset.shuffle(seed=seed)

    def _prepare_task_dataset(dataset: Dataset, task_type: str) -> Dataset:
        """Helper to transform a raw dataset for a specific task (encode/decode)."""

        def _transform_sample(sample):
            original_text = sample["text"]
            if task_type == "encode":
                prompt_content = ENCODE_PROMPT_TEMPLATE.format(text=original_text)
                return {
                    "prompt": [{"role": "user", "content": prompt_content}],
                    "answer": original_text,
                    "task": "encode",
                }
            elif task_type == "decode":
                encoded_text = base64.b64encode(original_text.encode("utf-8")).decode("utf-8")
                prompt_content = DECODE_PROMPT_TEMPLATE.format(encoded_text=encoded_text)
                return {
                    "prompt": [{"role": "user", "content": prompt_content}],
                    "answer": original_text,
                    "task": "decode",
                }
            return {}

        return dataset.map(_transform_sample, remove_columns=dataset.column_names)

    parser = Qwen3Parser() if use_think else vf.Parser()
    envs = []
    env_names = []

    if task is None or task == "encode":
        encode_rubric = vf.Rubric(funcs=[encode_similarity_reward], parser=parser)
        encode_env = vf.SingleTurnEnv(
            dataset=_prepare_task_dataset(raw_datasets["train"], "encode"),
            eval_dataset=_prepare_task_dataset(raw_datasets["test"], "encode"),
            rubric=encode_rubric,
            parser=parser,
            message_type="chat",
            **kwargs,
        )
        envs.append(encode_env)
        env_names.append("encode")

    if task is None or task == "decode":
        decode_rubric = vf.Rubric(funcs=[decode_similarity_reward], parser=parser)
        decode_env = vf.SingleTurnEnv(
            dataset=_prepare_task_dataset(raw_datasets["train"], "decode"),
            eval_dataset=_prepare_task_dataset(raw_datasets["test"], "decode"),
            rubric=decode_rubric,
            parser=parser,
            message_type="chat",
            **kwargs,
        )
        envs.append(decode_env)
        env_names.append("decode")

    if len(envs) == 1:
        return envs[0]
    else:
        return vf.EnvGroup(envs=envs, env_names=env_names)

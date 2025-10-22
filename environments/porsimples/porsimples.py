from collections.abc import Mapping

import evaluate
import pandas as pd
import verifiers as vf
from datasets import Dataset, DatasetDict
from easse.sari import corpus_sari
from verifiers.types import Messages

_rouge_metric = evaluate.load("rouge")
_bleu_metric = evaluate.load("bleu")
_meteor_metric = evaluate.load("meteor")


def sari_reward(completion: Messages, answer: str, info: dict, parser: vf.Parser) -> float:
    """
    Calculates the per-sample SARI score, the primary metric for text simplification.
    """
    try:
        # Use the parser to get the final assistant response string
        parsed_completion = parser.parse_answer(completion) or ""
        # Get the original complex sentence from the 'info' dict
        prompt_input = info.get("prompt_input", "")

        score = corpus_sari(orig_sents=[prompt_input], sys_sents=[parsed_completion], refs_sents=[[answer]])
        return float(score)
    except Exception:
        return 0.0


def rouge_1_reward(completion: Messages, answer: str, parser: vf.Parser) -> float:
    """Calculates ROUGE-1 score."""
    parsed_completion = parser.parse_answer(completion) or ""
    results = _rouge_metric.compute(predictions=[parsed_completion], references=[answer])
    return float(results["rouge1"])


def rouge_2_reward(completion: Messages, answer: str, parser: vf.Parser) -> float:
    """Calculates ROUGE-2 score."""
    parsed_completion = parser.parse_answer(completion) or ""
    results = _rouge_metric.compute(predictions=[parsed_completion], references=[answer])
    return float(results["rouge2"])


def rouge_L_reward(completion: Messages, answer: str, parser: vf.Parser) -> float:
    """Calculates ROUGE-L score."""
    parsed_completion = parser.parse_answer(completion) or ""
    results = _rouge_metric.compute(predictions=[parsed_completion], references=[answer])
    return float(results["rougeL"])


def rouge_Lsum_reward(completion: Messages, answer: str, parser: vf.Parser) -> float:
    """Calculates ROUGE-Lsum score."""
    parsed_completion = parser.parse_answer(completion) or ""
    results = _rouge_metric.compute(predictions=[parsed_completion], references=[answer])
    return float(results["rougeLsum"])


def bleu_reward(completion: Messages, answer: str, parser: vf.Parser) -> float:
    """Calculates BLEU score."""
    parsed_completion = parser.parse_answer(completion) or ""
    results = _bleu_metric.compute(predictions=[parsed_completion], references=[[answer]])
    return float(results["bleu"])


def meteor_reward(completion: Messages, answer: str, parser: vf.Parser) -> float:
    """Calculates METEOR score."""
    parsed_completion = parser.parse_answer(completion) or ""
    results = _meteor_metric.compute(predictions=[parsed_completion], references=[answer])
    return float(results["meteor"])


class Qwen3Parser(vf.Parser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def parse(self, text: str) -> str:
        if "</think>" not in text:
            return ""  # Cropped off before the </think> tag.
        return text.split("</think>")[1].strip()
    

def load_environment(seed: int = 3301, use_think: bool = False, **kwargs) -> vf.Environment:
    """
    Loads the PorSimples environment for Portuguese text simplification.
    This version manually replicates the dataset loading logic to work with modern
    versions of the `datasets` library, which no longer support remote loading scripts.
    """

    # --- Manual Dataset Loading ---
    _URLS = {
        "nat_str": "https://raw.githubusercontent.com/sidleal/porsimplessent/c0b7bb6ccda6b40ebd7f5524b08a5699b2266ffe/pss/pss2_align_length_nat_str.tsv",
        "ori_nat": "https://raw.githubusercontent.com/sidleal/porsimplessent/c0b7bb6ccda6b40ebd7f5524b08a5699b2266ffe/pss/pss2_align_length_ori_nat.tsv",
        "ori_str": "https://raw.githubusercontent.com/sidleal/porsimplessent/c0b7bb6ccda6b40ebd7f5524b08a5699b2266ffe/pss/pss2_align_length_ori_str.tsv",
    }

    df1 = pd.read_csv(_URLS["nat_str"], sep="\t", on_bad_lines="skip")
    df2 = pd.read_csv(_URLS["ori_nat"], sep="\t", on_bad_lines="skip")
    df3 = pd.read_csv(_URLS["ori_str"], sep="\t", on_bad_lines="skip")
    df = pd.concat([df1, df2, df3], axis=0)

    # Clean the data: Replace NaN values (floats) in text columns with empty strings.
    string_columns = ["level", "changed", "split", "sentence_text_from", "sentence_text_to"]
    for col in string_columns:
        df[col] = df[col].fillna("").astype(str)

    df = df.sort_values(by=["production_id", "sentence_text_from", "sentence_text_to"])
    prod_id_set = sorted(list(set(df["production_id"].values.tolist())))

    records = df.to_dict("records")
    processed_records = []
    for item in records:
        row = {
            key: item[key]
            for key in ["production_id", "level", "changed", "split", "sentence_text_from", "sentence_text_to"]
        }

        if item["changed"] == "S":
            if prod_id_set.index(item["production_id"]) % 2 == 0:
                row["sentence1"], row["sentence2"], row["label"] = (
                    item["sentence_text_from"],
                    item["sentence_text_to"],
                    2,
                )
            else:
                row["sentence1"], row["sentence2"], row["label"] = (
                    item["sentence_text_to"],
                    item["sentence_text_from"],
                    0,
                )
        else:
            row["sentence1"], row["sentence2"], row["label"] = item["sentence_text_from"], item["sentence_text_to"], 1

        processed_records.append(row)

    full_dataset = Dataset.from_list(processed_records)

    def add_mod_column(example):
        example["mod"] = example["production_id"] % 5
        return example

    full_dataset = full_dataset.map(add_mod_column)

    dataset = DatasetDict(
        {
            "train": full_dataset.filter(lambda x: x["mod"] not in [1, 2]),
            "validation": full_dataset.filter(lambda x: x["mod"] == 1),
            "test": full_dataset.filter(lambda x: x["mod"] == 2),
        }
    )
    # --- End Manual Dataset Loading ---

    def add_simpler_and_complexer_columns(example: Mapping) -> dict:
        if example["label"] == 0:
            example["simpler"] = example["sentence1"]
            example["complexer"] = example["sentence2"]
        elif example["label"] == 2:
            example["simpler"] = example["sentence2"]
            example["complexer"] = example["sentence1"]
        return example

    processed_dataset = (
        dataset.filter(lambda ex: ex["split"] == "N")
        .filter(lambda ex: ex["changed"] == "S")
        .filter(lambda ex: ex["level"] != "NAT->STR")
        .filter(lambda ex: ex["label"] != 1)
        .map(add_simpler_and_complexer_columns)
    )

    def format_for_verifiers(example: Mapping) -> dict:
        complex_sentence = example["complexer"]
        prompt_messages = [
            {
                "role": "user",
                "content": (
                    "Substitua a frase complexa por uma frase simples. "
                    "Mantenha o mesmo significado, mas torne-a mais simples. "
                    f"Frase complexa: {complex_sentence}\n"
                    "Responda apenas com a frase simplificada."
                ),
            }
        ]
        return {"prompt": prompt_messages, "answer": example["simpler"], "info": {"prompt_input": complex_sentence}}

    test_set = processed_dataset["test"]
    train_set = processed_dataset["train"]
    test_dataset = test_set.map(format_for_verifiers, remove_columns=test_set.column_names).shuffle(seed=seed)
    train_dataset = train_set.map(format_for_verifiers, remove_columns=train_set.column_names).shuffle(seed=seed)

    parser = Qwen3Parser() if use_think else vf.Parser()
    rubric = vf.Rubric(parser=parser)
    rubric.add_reward_func(sari_reward, weight=1.0)
    rubric.add_reward_func(rouge_1_reward, weight=0.0)
    rubric.add_reward_func(rouge_2_reward, weight=0.0)
    rubric.add_reward_func(rouge_L_reward, weight=0.0)
    rubric.add_reward_func(rouge_Lsum_reward, weight=0.0)
    rubric.add_reward_func(bleu_reward, weight=0.0)
    rubric.add_reward_func(meteor_reward, weight=0.0)

    return vf.SingleTurnEnv(eval_dataset=test_dataset, dataset=train_dataset, rubric=rubric, message_type="chat")

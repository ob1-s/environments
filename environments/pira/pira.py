from collections import Counter
from collections.abc import Mapping

import evaluate
import verifiers as vf
from datasets import load_dataset
from verifiers.types import Messages

# Pre-load metrics for efficiency, following the established pattern
_rouge_metric = evaluate.load("rouge")
_bleu_metric = evaluate.load("bleu")
_meteor_metric = evaluate.load("meteor")


def _calculate_overlap_metrics(prediction: str, reference: str) -> tuple[float, float, float]:
    """
    Core logic to calculate overlap-based precision, recall, and F1-score.
    This implementation is a direct port from the original brfauna evaluation script.
    """
    # Handle empty strings to avoid division by zero
    if not prediction or not reference:
        return 0.0, 0.0, 0.0

    pred_tokens = prediction.split()
    ref_tokens = reference.split()

    # If either list is empty after splitting, no overlap is possible.
    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0

    # Use Counter to find the multiset intersection of tokens
    common_tokens = Counter(pred_tokens) & Counter(ref_tokens)
    true_positives = sum(common_tokens.values())

    precision = true_positives / len(pred_tokens)
    recall = true_positives / len(ref_tokens)
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1_score


def overlap_f1_reward(completion: Messages, answer: str, parser: vf.Parser) -> float:
    """
    Calculates the Overlap F1-Score, the primary metric for the PIRA QA task.
    """
    try:
        parsed_completion = parser.parse_answer(completion) or ""
        _, _, f1 = _calculate_overlap_metrics(parsed_completion, answer)
        return float(f1)
    except Exception:
        return 0.0


def overlap_precision_reward(completion: Messages, answer: str, parser: vf.Parser) -> float:
    """Calculates the Overlap Precision score for logging."""
    try:
        parsed_completion = parser.parse_answer(completion) or ""
        precision, _, _ = _calculate_overlap_metrics(parsed_completion, answer)
        return float(precision)
    except Exception:
        return 0.0


def overlap_recall_reward(completion: Messages, answer: str, parser: vf.Parser) -> float:
    """Calculates the Overlap Recall score for logging."""
    try:
        parsed_completion = parser.parse_answer(completion) or ""
        _, recall, _ = _calculate_overlap_metrics(parsed_completion, answer)
        return float(recall)
    except Exception:
        return 0.0


# --- Standard Metrics (Identical to porsimples) ---


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
    Loads the PIRA environment for Portuguese generative question-answering.
    The task is to answer questions about oceans, the Brazilian coast, and climate change.
    """
    # Load the dataset directly from the Hugging Face Hub.
    dataset = load_dataset("paulopirozelli/pira")

    def format_for_verifiers(example: Mapping) -> dict:
        """
        Formats a raw dataset example into the structure required by verifiers.
        The prompt is identical to the one used in the brfauna codebase.
        """
        question = example["question_pt_origin"]
        prompt_content = (
            "Responda à seguinte pergunta com base em seu conhecimento geral sobre oceanos, "
            "a costa brasileira e as mudanças climáticas.\n"
            "Seja breve e objetivo.\n"
            f"Pergunta: {question}"
        )

        prompt_messages = [{"role": "user", "content": prompt_content}]

        return {
            "prompt": prompt_messages,
            "answer": example["answer_pt_origin"],
            "info": {},  # No extra info needed for these metrics
        }

    # Prepare the train and evaluation datasets
    test_set = dataset["test"]
    train_set = dataset["train"]

    eval_dataset = test_set.map(format_for_verifiers, remove_columns=test_set.column_names).shuffle(seed=seed)
    train_dataset = train_set.map(format_for_verifiers, remove_columns=train_set.column_names).shuffle(seed=seed)

    # Configure the rubric with all metrics
    parser = Qwen3Parser() if use_think else vf.Parser()
    rubric = vf.Rubric(parser=parser)

    # Set the primary metric for the environment's reward signal
    rubric.add_reward_func(overlap_f1_reward, weight=1.0)

    # Add all other metrics for comprehensive logging, with zero weight
    rubric.add_reward_func(overlap_precision_reward, weight=0.0)
    rubric.add_reward_func(overlap_recall_reward, weight=0.0)
    rubric.add_reward_func(rouge_1_reward, weight=0.0)
    rubric.add_reward_func(rouge_2_reward, weight=0.0)
    rubric.add_reward_func(rouge_L_reward, weight=0.0)
    rubric.add_reward_func(rouge_Lsum_reward, weight=0.0)
    rubric.add_reward_func(bleu_reward, weight=0.0)
    rubric.add_reward_func(meteor_reward, weight=0.0)

    print(f"eval_dataset: {len(eval_dataset)} examples")
    print(f"train_dataset: {len(train_dataset)} examples")

    return vf.SingleTurnEnv(
        eval_dataset=eval_dataset,
        dataset=train_dataset,
        rubric=rubric,
    )
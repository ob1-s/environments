# porsimples

### Overview
- **Environment ID**: `porsimples`
- **Short description**: Portuguese text simplfication task.
- **Tags**: text-simplification, portuguese, single-turn, train, eval

### Datasets
- **Primary dataset(s)**: PorSimplesSent - A Portuguese corpus of aligned sentence pairs to investigate sentence readability assessment
- **Source links**: [dataset](https://github.com/sidleal/porsimplessent), [eval/research](https://github.com/MeLLL-UFF/brfauna-gen-eval)
- **Split sizes**: `test`: 349, `train`: 1100.

### Task
- **Type**: single-turn
- **Parser**: `vf.Parser` if `use_think=False` (default), else a custom `Qwen3Parser`
- **Rubric overview**: The rubric calculates multiple standard text generation metrics. The primary reward is the **per-sample SARI score**, which is specifically designed for evaluating text simplification. Other metrics like ROUGE, BLEU, and METEOR are also calculated for comprehensive analysis but are not weighted in the final reward by default.

    - **Note**:
    For this version, I decided to not include the bert_score and corpus-level scores used in the original research "Exploring Brazil's LLM Fauna" for simplicity.

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval porsimples
```

Configure model and sampling:

```bash
uv run vf-eval porsimples   -m gpt-4.1-mini   -n 20 -r 3 -t 1024 -T 0.7   -a '{"seed": 42}'  # env-specific args as JSON
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `seed` | int | `3301` | Random seed for dataset shuffling |
| `use_think` | bool | `False` | Whether to use the qwen3 parser. Set to `true` for reasoning models which output their CoT, else set to `false`|

### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `reward` | The main scalar reward. By default, this is equal to the `sari_reward`. |
| `sari_reward` | **(Primary Metric)** Measures the quality of a simplification by comparing n-grams that are added, deleted, and kept against the reference and original sentences. A higher score is better. Weighed at 1.0. |
| `rouge_1_reward` | ROUGE-1 score, measuring the overlap of unigrams (single words) between the generated and reference text. Weighed at 0.0. |
| `rouge_2_reward` | ROUGE-2 score, measuring the overlap of bigrams (pairs of words). Weighed at 0.0. |
| `rouge_L_reward` | ROUGE-L score, based on the longest common subsequence between the texts. Weighed at 0.0. |
| `rouge_Lsum_reward`| ROUGE-Lsum score, which is a summary-level version of ROUGE-L. Weighed at 0.0. |
| `bleu_reward` | BLEU score, a precision-focused metric that measures n-gram overlap. It indicates how much of the generated text appears in the reference. Weighed at 0.0. |
| `meteor_reward` | METEOR score, which aligns words based on exact matches, stems, and synonyms for a more robust similarity measure than BLEU. Weighed at 0.0. |

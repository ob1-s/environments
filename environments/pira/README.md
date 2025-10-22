# pira

### Overview
- **Environment ID**: `pira`
- **Short description**: Portuguese generative question-answering about oceans, the Brazilian coast, and climate change.
- **Tags**: question-answering, portuguese, single-turn, train, eval, science

### Datasets
- **Primary dataset(s)**: Pirá - A bilingual (Portuguese/English) dataset for question-answering about the Ocean, the Brazilian coast, and climate change.
- **Source links**: [dataset](https://huggingface.co/datasets/paulopirozelli/pira), [eval/research](https://github.com/MeLLL-UFF/brfauna-gen-eval)
- **Split sizes**: `eval`: 227, `train`: 1806.

### Task
- **Type**: single-turn
- **Parser**: `vf.Parser` if `use_think=False` (default), else `Qwen3Parser`
- **Rubric overview**: The rubric calculates multiple standard text generation metrics. The primary reward is the **Overlap F1-Score**, which measures the lexical overlap between the generated and reference answers. Other metrics like ROUGE, BLEU, and METEOR are also calculated for comprehensive analysis but are not weighted in the final reward by default.
    - **Note**: The original research uses a custom BERTScore metric, which is not included in this version for simplicity.

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval pira
```

Configure model and sampling:

```bash
uv run vf-eval pira   -m gpt-4.1-mini   -n 20 -r 3 -t 1024 -T 0.7   -a '{"seed": "42"}'  # env-specific args as JSON
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments
Document any supported environment arguments and their meaning. Example:

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `seed` | int | `3301` | Random seed for dataset shuffling |
| `use_think` | bool | `False` | Whether to use the qwen3 parser. Set to `true` for reasoning models which output their CoT, else set to `false`|

### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `reward` | The main scalar reward. By default, this is equal to the `overlap_f1_reward`. |
| `overlap_f1_reward` | **(Primary Metric)** The harmonic mean of precision and recall based on the multiset intersection of tokens between the generated and reference answers. A higher score is better. Weighed at 1.0. |
| `overlap_precision_reward` | The precision component of the F1 score, measuring what fraction of the generated tokens are relevant. Weighed at 0.0. |
| `overlap_recall_reward` | The recall component of the F1 score, measuring what fraction of the reference tokens were generated. Weighed at 0.0. |
| `rouge_1_reward` | ROUGE-1 score, measuring the overlap of unigrams (single words) between the generated and reference text. Weighed at 0.0. |
| `rouge_2_reward` | ROUGE-2 score, measuring the overlap of bigrams (pairs of words). Weighed at 0.0. |
| `rouge_L_reward` | ROUGE-L score, based on the longest common subsequence between the texts. Weighed at 0.0. |
| `rouge_Lsum_reward`| ROUGE-Lsum score, which is a summary-level version of ROUGE-L. Weighed at 0.0. |
| `bleu_reward` | BLEU score, a precision-focused metric that measures n-gram overlap. It indicates how much of the generated text appears in the reference. Weighed at 0.0. |
| `meteor_reward` | METEOR score, which aligns words based on exact matches, stems, and synonyms for a more robust similarity measure than BLEU. Weighed at 0.0. |

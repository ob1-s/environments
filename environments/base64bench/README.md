# base64bench

> Based on the original experiment [Base64Bench- How good are LLMs at base64, and why care about it?](https://www.lesswrong.com/posts/5F6ncBfjh2Bxnm6CJ/base64bench-how-good-are-llms-at-base64-and-why-care-about) by Rich Barton-Cooper

### Overview
- **Environment ID**: `base64bench`
- **Short description**: Evaluates a model's ability to accurately perform base64 encoding and decoding across a variety of text and data formats.
- **Tags**: `base64`, `encoding`, `decoding`, `eval`, `train`

### Datasets
- **Primary dataset(s)**: The environment uses pre-generated `train.jsonl` and `eval.jsonl` files containing diverse data types, such as UUIDs, API keys, JSON, natural language, and cryptographic data representations.
- **Source links**: The datasets were generated with https://github.com/richcooper95/base-64-bench
- **Split sizes**: 340 test examples (10 per data type); 1360 train examples (40 per data type). Each example is used twice, once for an encoding task and once for a decoding task.

### Task
- **Type**: single-turn
- **Parser**: `vf.Parser` if `use_think=False` (default), else a custom `Qwen3Parser` for models that use `<think>` tags.
- **Rubric overview**: The reward is calculated using normalized Levenshtein similarity. For encoding tasks, the model's output is decoded and compared to the original text. For decoding tasks, the model's output is directly compared to the original text.

### Quickstart
Run an evaluation with default settings on both encode and decode tasks:

```bash
uv run vf-eval base64bench
```

Configure the model, limit the evaluation to the `decode` task, and limit the evaluation set with 3 examples from each data type:

```bash
uv run vf-eval base64bench \
  -m gpt-4.1-mini \
  -n -1 \
  -a '{"task": "decode", "examples_per_data_type": 3}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `train_file` | str | `"train.jsonl"` | The name of the training dataset file located in the environment's directory. |
| `eval_file` | str | `"eval.jsonl"` | The name of the evaluation dataset file located in the environment's directory. |
| `seed` | int | `3301` | The seed used for shuffling the dataset. |
| `use_think` | bool | `False` | Whether to use the qwen3 parser. Set to `true` for reasoning models which output their CoT, else set to `false`. |
| `task` | str | `None` | Restricts the evaluation to a specific task. Can be `"encode"`, `"decode"`, or `None` (default) to run both. |
| `examples_per_data_type`| int | `None` | If set, creates a balanced evaluation set by sampling N examples of each data type from the eval file. |

### Metrics
The primary reward is the similarity score itself. The metrics are named based on the task.

| Metric | Meaning |
| ------ | ------- |
| `reward` | The main scalar reward. By default, this is a sum of the encode and decode rewards. |
| `encode_similarity_reward` | Normalized Levenshtein similarity between the original text and the model's base64 output after it has been decoded. A score of 1.0 is a perfect match. |
| `decode_similarity_reward` | Normalized Levenshtein similarity between the original text and the model's decoded output. A score of 1.0 is a perfect match. |
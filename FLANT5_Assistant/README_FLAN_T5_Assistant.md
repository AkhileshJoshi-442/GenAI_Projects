# FLAN-T5 Document Summarizer & Q&A Assistant

## Description

A local, CPU-friendly NLP tool built on Google's FLAN-T5 (an instruction-tuned
Seq2Seq language model) that performs two tasks: abstractive text summarization
and closed-context question answering from a user-provided document. The model
runs entirely on-device with no API key or GPU required.

## Features

- Summarizes pasted text into concise bullet points using FLAN-T5
- Answers questions strictly from a user-provided `context.txt` file
- Returns "Not found" when the answer is absent from the context, preventing
  hallucinated responses
- Evaluates summarization output using word-count compression ratio and a
  ROUGE-1 extractiveness score
- Runs fully offline on CPU — no internet connection needed after model download
- Simple interactive CLI with three options: summarize, Q&A, and exit

## Requirements

### Standard library (included with Python, no installation needed)

- `os`

### Third-party (must be installed)

- `transformers`
- `evaluate`
- `torch`
- `rouge_score`

Install all third-party libraries with:

```
pip install transformers evaluate torch rouge_score
```

## Usage

```
python flan_t5_assistant.py
```

The script launches an interactive menu. No command line arguments are needed.

```
----------
FLAN-T5 Summarizer & Q&A Assistant
1. Summarize text
2. Questions & Answers over local context.txt
0. Exit
----------
```

### Option 1 — Summarize text

Select `1`, paste your text, and press Enter on a blank line to submit. The
model generates a bullet-point summary and prints compression metrics.

### Option 2 — Question & Answers

Place your reference document content in a file named `context.txt` in the
same folder as the script. Select `2`, type your question, and the model
answers using only that file. If the answer is not present in the context,
the model replies exactly: `Not found.`

## Function Reference

### `run_flan(prompt, max_new_tokens=128)`
Tokenizes the input prompt, runs the FLAN-T5 model with sampling parameters
(`top_p=0.9`, `temperature=0.7`), and returns the decoded output string with
special tokens removed. This is the core inference function used by both
`summarize_text` and `answer_from_context`.

### `summarize_text(text)`
Builds a single-instruction summarization prompt from the input text and calls
`run_flan` with a higher token budget (`max_new_tokens=200`). Returns the
model's generated summary.

### `load_context(path="context.txt")`
Reads and returns the full contents of the file at `path`. Returns an empty
string if the file does not exist, which is handled gracefully in `main`.

### `answer_from_context(question, context)`
Constructs a strict grounded-QA prompt that instructs the model to answer
only from the provided context and to reply `Not found.` if the answer is
absent. Calls `run_flan` and returns the model's answer.

### `evaluate_performance(original_text, summary_text)`
Computes two metrics for the summarization output: word-count compression
ratio (percentage of words removed) and a ROUGE-1 score. The ROUGE-1 score
here compares the summary against the source text rather than a human
reference, so it measures extractiveness — how much of the original wording
was retained — not summary quality. Returns a dictionary with both values, or
`None` if the original text is empty.

### `main()`
Prints the interactive menu and runs a loop that reads the user's choice and
calls the appropriate function. Handles empty input and invalid choices
without crashing.

## Notes and Limitations

- **Model size**: `google/flan-t5-small` (~80M parameters, ~300 MB) is
  downloaded automatically on first run via the HuggingFace Hub and cached
  locally. Subsequent runs load from cache with no download.
- **Context window**: FLAN-T5 has a 512-token input limit. Long documents
  passed to Option 2 will be silently truncated. Keep `context.txt` under
  approximately 350 words for reliable results.
- **Summarization quality**: FLAN-T5-small is a lightweight model. Output
  quality improves significantly with `google/flan-t5-base` (~250M parameters)
  at the cost of slower CPU inference. Change `MODEL_NAME` in the script to
  switch.
- **ROUGE interpretation**: The ROUGE-1 score displayed after summarization
  measures how much of the original wording appears in the summary. A high
  score means the summary copies phrases from the source; it does not mean the
  summary is accurate or useful.
- **Offline use**: After the first download, the model runs with no internet
  connection. If you move the script, the HuggingFace cache remains in your
  home directory (`~/.cache/huggingface`).
- Stop the script at the menu prompt with `Ctrl+C`. This will print a
  `KeyboardInterrupt` message, which is expected.

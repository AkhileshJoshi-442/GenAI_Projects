import os
# Disable parallelism to avoid potential deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import from Hugging Face Transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from evaluate import load

# Choose an instruction-tuned model
MODEL_NAME = "google/flan-t5-small"
# A lightweight version of FLAN-T5
# About 80 million parameters

print(f"FLAN-T5 Summarizer & Q&A Assistant — loading {MODEL_NAME}...")

# Load tokenizer and sequence-to-sequence model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def run_flan(prompt: str, max_new_tokens: int = 128) -> str:
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    # Generate text with light sampling for naturalness
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )
    # Decode token IDs back into a clean string
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def summarize_text(text: str) -> str:
    # Instructs the model to produce a concise abstractive summary
    prompt = (
        f"Summarize the following text in 4-6 bullet points. "
        f"Include all specific technical terms, dates, and names from the text:\n\n{text}"
    )
    return run_flan(prompt, max_new_tokens=200)


def load_context(path: str = "context.txt") -> str:
    """Loads content from a local text file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def answer_from_context(question: str, context: str) -> str:
    """Answers questions using ONLY the provided context."""
    if not context.strip():
        return "Context file not found or empty. Create 'context.txt' first."

    # Construct a strict prompt for FLAN-T5
    prompt = (
        "You are a helpful assistant. Answer the question ONLY using the context.\n"
        "If the answer is not in the context, reply exactly: Not found.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )
    return run_flan(prompt, max_new_tokens=120)


def evaluate_performance(original_text, summary_text):
    """
    Computes word-count compression ratio and ROUGE-1 score.
    Note: ROUGE here compares summary against the source text (not a
    human reference), so it measures extractiveness rather than quality.
    Use compression ratio as the primary metric.
    """
    # Load the ROUGE metric
    rouge = load("rouge")

    # Calculate ROUGE-1 (extractiveness proxy, not a quality score)
    results = rouge.compute(predictions=[summary_text], references=[original_text])

    # Calculate compression ratio
    original_words = len(original_text.split())
    summary_words = len(summary_text.split())

    # Avoid division by zero
    if original_words == 0:
        return None

    reduction = ((original_words - summary_words) / original_words) * 100

    return {
        "rouge1_extractiveness": results["rouge1"],
        "volume_reduction": reduction
    }


def main():
    """Entry point function for the user interface."""
    print("-" * 10)
    print("FLAN-T5 Summarizer & Q&A Assistant")
    print("1. Summarize text")
    print("2. Questions & Answers over local context.txt")
    print("0. Exit")
    print("-" * 10)

    while True:

        choice = input("\nChoose an option (1/2/0): ").strip()

        if choice == "0":
            print("Exiting FLAN-T5 Assistant.")
            break

        elif choice == "1":
            print("You have selected Summarisation option...")
            print("\nPaste text to summarize. End with a blank line:")
            lines = []
            while True:
                line = input()
                if not line.strip():
                    break
                lines.append(line)

            text = "\n".join(lines).strip()
            if not text:
                print("No text received.")
                continue

            print("\nSummary:")
            summary = summarize_text(text)
            print(summary)

            metrics = evaluate_performance(text, summary)
            print(f"\n--- Performance Metrics ---")
            print(f"Text Volume Reduction: {metrics['volume_reduction']:.2f}%")
            print(f"ROUGE-1 (extractiveness proxy): {metrics['rouge1_extractiveness']:.4f}")

        elif choice == "2":
            ctx = load_context("context.txt")
            if not ctx.strip():
                print("Missing 'context.txt'. Create it in the same folder and try again.")
                continue

            q = input("\nAsk a question about your context: ").strip()
            if not q:
                print("No question received.")
                continue

            print("\nAnswer:")
            print(answer_from_context(q, ctx))

        else:
            print("Please choose 1, 2, or 0.")


if __name__ == "__main__":
    main()

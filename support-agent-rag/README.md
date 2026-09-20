# Support Agent (RAG + Tool Calling)

## What it does
A customer-support agent built with Python and the Gemini API. It answers policy questions from a small document set (RAG), does math with a calculator tool, and says "I don't have that information" when the answer isn't in the documents.

## How it works
1. The user asks a question.
2. The LLM decides which tool to use: `search_docs` or `calculator`.
3. My Python code runs the tool (the LLM never runs code itself).
4. The result goes back to the LLM, which writes the final answer.

`search_docs` turns each document and the question into embeddings, picks the closest document by cosine similarity, and returns "Not found" if the score is below a threshold (0.6).

## Problems I hit and how I fixed them
- Free-tier rate limit (429): added retries with waiting.
- Agent returned "None" when the model wanted a second tool call: replaced the single call with a loop capped at 4 steps.
- Daily quota exhausted: learned quotas are per model and per project, and switched models.
- A test failed although the agent was right: my test checked exact words, so I widened it to check meaning.
- A tool crash (division by zero) is caught and passed back to the LLM, so the agent explains it instead of crashing.

## How I test it
`eval.py` runs 8 fixed questions and checks both the tool chosen and the answer. I re-run it after every change to catch regressions.

## What I'd improve next
- Load real documents (PDF/text files) and split them into chunks
- Add a vector database
- Rebuild the loop in LangGraph
- Use an LLM as a judge for evals
- Connect it to WhatsApp or voice

## Run it
1. `pip install google-genai python-dotenv`
2. Create a `.env` file with `GEMINI_API_KEY=your_key`
3. `python agent.py` (demo) or `python eval.py` (tests)
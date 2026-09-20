import time
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors
from rag import search_docs

load_dotenv()
client = genai.Client()
MODEL = "gemini-3.5-flash-lite"
MAX_STEPS = 4        # the agent may use tools at most 4 times per question
TOOL_LOG = []        # remembers which tools were used (for the tests)

# ---------- TOOLS ----------
def calculator(expression: str) -> str:
    """Evaluate a math expression like '15/100*2480'."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:      # never eval raw text blindly
        return "Error: only numbers and + - * / ( ) are allowed"
    return str(eval(expression))

TOOLS = {"calculator": calculator, "search_docs": search_docs}

SYSTEM_PROMPT = (
    "You are a customer support assistant. "
    "For any question about company policies, ALWAYS call search_docs and answer "
    "only from what it returns. If it returns 'Not found', do not search again; "
    "just say you don't have that information. Use calculator for any math. "
    "If a tool returns an error, explain the problem simply to the user. "
    "Keep answers short."
)

config = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    tools=[calculator, search_docs],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

# ---------- FAILURE HANDLING ----------
def call_llm(contents, retries=3):
    """Call the LLM. If rate-limited or the server is busy, wait and retry."""
    for attempt in range(1, retries + 1):
        try:
            return client.models.generate_content(model=MODEL, contents=contents, config=config)
        except errors.APIError as e:
            if e.code in (429, 500, 503) and attempt < retries:
                wait = 15 * attempt
                print(f"[retry] error {e.code}, waiting {wait}s (attempt {attempt}/{retries})")
                time.sleep(wait)
            else:
                raise

def run_tool(name, args):
    """Run a tool safely. Never crash; return a clear error message instead."""
    TOOL_LOG.append(name)
    if name not in TOOLS:
        return f"Error: unknown tool '{name}'"
    try:
        return TOOLS[name](**args)
    except Exception as e:
        return f"Error: tool failed ({e})"

# ---------- THE AGENT LOOP ----------
def _ask(question: str) -> str:
    contents = [types.Content(role="user", parts=[types.Part(text=question)])]

    for step in range(MAX_STEPS):
        response = call_llm(contents)

        if not response.function_calls:                 # no tool wanted -> final answer
            return response.text or "Sorry, I couldn't produce an answer."

        contents.append(response.candidates[0].content)  # the LLM's tool request(s)
        result_parts = []
        for call in response.function_calls:
            args = dict(call.args or {})
            print(f"[step {step + 1}] LLM chose tool: {call.name}({args})")
            result = run_tool(call.name, args)           # OUR code runs the tool
            print(f"[tool result] {result}")
            result_parts.append(
                types.Part.from_function_response(name=call.name, response={"result": result})
            )
        contents.append(types.Content(role="user", parts=result_parts))

    return "Sorry, I couldn't finish that request. Please try rephrasing it."

def ask(question: str) -> str:
    """Last safety net: if everything fails, give a friendly message."""
    try:
        return _ask(question)
    except Exception as e:
        print(f"[error] {e}")
        return "Sorry, I'm having trouble right now. Please try again in a moment."

if __name__ == "__main__":
    questions = [
        "How long do refunds take?",
        "Do you sell laptops on EMI?",
        "What is 10 divided by 0?",
    ]
    for q in questions:
        print("\nQ:", q)
        print("A:", ask(q))
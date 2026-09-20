import sys
import time
import agent

PAUSE = 30   # seconds between tests, to stay under the free-tier limit

# Each test: the question, the tool we EXPECT (None = no tool),
# and words of which the answer must contain at least one.
TESTS = [
    {"name": "refund time",    "q": "How long do refunds take?",
     "tool": "search_docs", "any_of": ["7"]},
    {"name": "support hours",  "q": "When is your support team available?",
     "tool": "search_docs", "any_of": ["saturday", "9 am", "9am"]},
    {"name": "express cost",   "q": "How much does express shipping cost?",
     "tool": "search_docs", "any_of": ["199"]},
    {"name": "password reset", "q": "How do I reset my password?",
     "tool": "search_docs", "any_of": ["forgot"]},
    {"name": "unknown topic",  "q": "Do you sell laptops on EMI?",
     "tool": "search_docs", "any_of": ["don't have", "do not have", "not have"]},
    {"name": "math",           "q": "What is 15% of 2480?",
     "tool": "calculator",  "any_of": ["372"]},
    {"name": "math error",     "q": "What is 10 divided by 0?",
     "tool": "calculator",  "any_of": ["zero", "undefined", "cannot", "not possible",
                                       "impossible", "not allowed"]},
    {"name": "small talk",     "q": "Say hello in one line.",
     "tool": None,          "any_of": ["hello", "hi"]},
]

def run_test(t):
    agent.TOOL_LOG.clear()
    answer = agent.ask(t["q"])
    used = agent.TOOL_LOG[0] if agent.TOOL_LOG else None
    tool_ok = (used == t["tool"])
    answer_ok = any(word in answer.lower() for word in t["any_of"])
    return tool_ok, answer_ok, used, answer

if __name__ == "__main__":
    # Optional: run only tests whose name contains the given text
    only = sys.argv[1] if len(sys.argv) > 1 else None
    tests = [t for t in TESTS if only is None or only in t["name"]]

    passed = 0
    for i, t in enumerate(tests):
        tool_ok, answer_ok, used, answer = run_test(t)
        ok = tool_ok and answer_ok
        passed += ok
        print(f"\n[{'PASS' if ok else 'FAIL'}] {t['name']}")
        print(f"   answer: {answer}")
        if not ok:
            print(f"   expected tool: {t['tool']} | used: {used}")
        if i < len(tests) - 1:
            time.sleep(PAUSE)
    print(f"\n===== SCORE: {passed}/{len(tests)} passed =====")
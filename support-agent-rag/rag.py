import math
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()
EMBED_MODEL = "gemini-embedding-001"
THRESHOLD = 0.6      # below this score, we say "Not found" (you'll tune this)

# Our tiny "company wiki": each string is one chunk
DOCS = [
    "Refund policy: Customers can request a refund within 14 days of purchase. Refunds are processed in 5 to 7 business days to the original payment method.",
    "Shipping: Orders ship within 2 business days. Standard delivery takes 4 to 6 days. Express delivery takes 1 to 2 days and costs 199 rupees extra.",
    "Support hours: Our support team is available Monday to Saturday, 9 AM to 6 PM IST. There is no support on Sundays. Email: help@example.com",
    "Warranty: All electronic products come with a 1 year warranty. Damage caused by water or drops is not covered.",
    "Account: To reset your password, click 'Forgot password' on the login page and follow the email link. The link expires in 30 minutes.",
]

def embed(text):
    """Turn text into its 'meaning fingerprint' (a list of numbers)."""
    result = client.models.embed_content(model=EMBED_MODEL, contents=text)
    return result.embeddings[0].values

def similarity(a, b):
    """How close are two fingerprints? 1.0 = identical meaning."""
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

# Fingerprint every document once, when the program starts
DOC_VECTORS = [embed(d) for d in DOCS]

def search_docs(query: str) -> str:
    """Search the company documents and return the most relevant text."""
    q = embed(query)
    scored = sorted(
        ((similarity(q, v), d) for v, d in zip(DOC_VECTORS, DOCS)),
        reverse=True,
    )
    print(f"[retrieval] best score: {scored[0][0]:.2f}")
    if scored[0][0] < THRESHOLD:
        return "Not found"
    return scored[0][1]
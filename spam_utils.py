import re

# Stop words: NLTK if available, otherwise scikit-learn's built-in list
try:
    import nltk
    try:
        from nltk.corpus import stopwords
        STOP_WORDS = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        STOP_WORDS = set(stopwords.words("english"))
except Exception:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    STOP_WORDS = set(ENGLISH_STOP_WORDS)

URL_RE   = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\S+@\S+")
PHONE_RE = re.compile(r"\b\d{5,}\b")
NUM_RE   = re.compile(r"\b\d+\b")
TOKEN_RE = re.compile(r"[a-z]+(?:\'[a-z]+)?|<[a-z]+>")

def clean_text(text: str) -> str:
    text = str(text).lower()                    # lowercasing
    text = URL_RE.sub(" <url> ", text)
    text = EMAIL_RE.sub(" <email> ", text)
    text = PHONE_RE.sub(" <phone> ", text)
    text = NUM_RE.sub(" <num> ", text)
    text = text.replace("£", " <money> ").replace("$", " <money> ")
    return text

def tokenize(text: str):
    """Regex tokenization + stop-word removal."""
    tokens = TOKEN_RE.findall(clean_text(text))
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

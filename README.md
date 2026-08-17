# Negation Blindness

Embedding models score a sentence and its own negation as near-identical — so vector
search and RAG retrieval are effectively blind to "not".

Work in progress. Measured numbers, charts, and the full writeup land as the experiment
is built out.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

Python 3.10+. Core experiment runs locally on CPU — no API keys required.

## License

MIT

# TikTok Streamlit App

This app wraps your TikTok analysis flow into a Streamlit interface with two sections:

- **Analysis by Post** (primary flow)
- **Analysis by Search**

## Files included

- `app.py` — Streamlit UI
- `services.py` — TikTok fetch, Gemini analysis, comments processing, payload prep, API send
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — Streamlit theme config
- `.streamlit/secrets.toml.example` — example secrets file

## Setup

```bash
pip install -r requirements.txt
```

Create a secrets file:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then fill in your real keys in `.streamlit/secrets.toml`.

## Run

```bash
streamlit run app.py
```

## Notes

- I corrected the search-mode condition from `elif postURLs:` to `elif searchQueries:`.
- The app uses **Analysis by Post** as the first/default tab.
- The UI uses your requested main color, with a solid fallback in Streamlit theme config where alpha hex is not supported.

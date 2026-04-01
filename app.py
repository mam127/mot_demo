import json
from typing import Optional

import pandas as pd
import streamlit as st

from services import (
    DATE_RANGE_MAP,
    clean_data,
    load_analyze_tiktok_video,
    parse_lines,
    prepare_payload,
    send_to_api,
)


PRIMARY_COLOR = "#051f1fcc"
PRIMARY_SOLID = "#051f1f"
DEFAULT_PROJECT_ID = "mot-saudi"
DEFAULT_API_URL = "https://public-api.anecdoteai.com/inject"


st.set_page_config(
    page_title="TikTok Analysis App",
    page_icon="🎥",
    layout="wide",
)


st.markdown(
    f"""
    <style>
        :root {{
            --brand: {PRIMARY_COLOR};
            --brand-solid: {PRIMARY_SOLID};
        }}

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}

        h1, h2, h3 {{
            color: var(--brand-solid);
        }}

        .stTabs [data-baseweb="tab"] {{
            font-weight: 700;
        }}

        .stTabs [aria-selected="true"] {{
            color: var(--brand-solid) !important;
            border-bottom-color: var(--brand-solid) !important;
        }}

        .stButton > button,
        .stDownloadButton > button {{
            background: var(--brand);
            color: white;
            border: 1px solid var(--brand-solid);
            border-radius: 10px;
            font-weight: 600;
        }}

        .stButton > button:hover,
        .stDownloadButton > button:hover {{
            border-color: var(--brand-solid);
            color: white;
        }}

        .app-banner {{
            padding: 1rem 1.2rem;
            border-radius: 14px;
            background: rgba(5, 31, 31, 0.08);
            border: 1px solid rgba(5, 31, 31, 0.18);
            margin-bottom: 1rem;
        }}

        .section-card {{
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid rgba(5, 31, 31, 0.12);
            background: rgba(255, 255, 255, 0.7);
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def dataframe_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


@st.cache_data(show_spinner=False)
def dict_to_json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")



def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass
    return default



def render_results(state_key: str, api_url: str, api_headers: dict, project_id: str):
    result = st.session_state.get(state_key)
    if not result:
        return

    raw_df = result["raw_df"]
    cleaned_df = result["cleaned_df"]
    payload = result["payload"]

    st.success(f"Analysis completed for {len(raw_df)} video(s).")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Videos", len(raw_df))
    metric_cols[1].metric("Total Views", int(pd.to_numeric(raw_df.get("playCount"), errors="coerce").fillna(0).sum()))
    metric_cols[2].metric("Total Likes", int(pd.to_numeric(raw_df.get("diggCount"), errors="coerce").fillna(0).sum()))
    metric_cols[3].metric("Total Comments", int(pd.to_numeric(raw_df.get("commentCount"), errors="coerce").fillna(0).sum()))

    st.subheader("Processed Data")
    st.dataframe(cleaned_df, use_container_width=True)

    with st.expander("Preview payload JSON", expanded=False):
        st.json(payload)

    with st.expander("Preview raw analysis output", expanded=False):
        preview_columns = [
            col
            for col in [
                "webVideoUrl",
                "text",
                "spoken_content_analysis",
                "visual_content_analysis",
                "audience",
                "sentiment_analysis",
                "transcript",
                "comments_summary",
                "comments_txt",
            ]
            if col in raw_df.columns
        ]
        st.dataframe(raw_df[preview_columns], use_container_width=True)

    download_cols = st.columns(2)
    download_cols[0].download_button(
        "Download processed CSV",
        data=dataframe_to_csv(cleaned_df),
        file_name="tiktok_processed_data.csv",
        mime="text/csv",
        use_container_width=True,
    )
    download_cols[1].download_button(
        "Download payload JSON",
        data=dict_to_json_bytes(payload),
        file_name="tiktok_payload.json",
        mime="application/json",
        use_container_width=True,
    )

    st.subheader("Send to API")
    st.caption(f"Project ID: {project_id}")
    if st.button(f"Send latest result to API ({state_key})", use_container_width=True):
        ok, status_code, body = send_to_api(payload, api_url=api_url, api_headers=api_headers)
        if ok:
            st.success(f"Sent successfully. Status code: {status_code}")
            st.json(body)
        else:
            st.error(f"Request failed. Status code: {status_code}")
            st.code(str(body))


st.title("TikTok Analysis App")
st.markdown(
    """
    <div class="app-banner">
        Analyze TikTok videos either by direct post URL or by search query, then transform the output into the ingestion payload.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Configuration")

    gemini_api_key = st.text_input(
        "GEMINI_API_KEY",
        value=get_secret("GEMINI_API_KEY", ""),
        type="password",
    )
    apify_api_token = st.text_input(
        "APIFY_API_TOKEN",
        value=get_secret("APIFY_API_TOKEN", ""),
        type="password",
    )
    api_bearer = st.text_input(
        "ANECDOTE_API_TOKEN",
        value=get_secret("ANECDOTE_API_TOKEN", ""),
        type="password",
    )

    project_id = st.text_input("Project ID", value=get_secret("PROJECT_ID", DEFAULT_PROJECT_ID))
    api_url = st.text_input("API URL", value=get_secret("API_URL", DEFAULT_API_URL))
    max_workers = st.slider("Parallel workers", min_value=1, max_value=12, value=4)

    st.markdown("---")
    st.caption("You can keep credentials in `.streamlit/secrets.toml` or enter them here.")

api_headers = {"Authorization": f"Bearer {api_bearer}"} if api_bearer else {}

missing = []
if not gemini_api_key:
    missing.append("GEMINI_API_KEY")
if not apify_api_token:
    missing.append("APIFY_API_TOKEN")

if missing:
    st.warning("Missing required configuration: " + ", ".join(missing))

post_tab, search_tab = st.tabs(["Analysis by Post", "Analysis by Search"])

with post_tab:
    st.subheader("Analysis by Post")
    st.caption("Primary flow. Add one TikTok post URL per line.")

    post_urls_text = st.text_area(
        "Post URLs",
        height=180,
        placeholder="https://www.tiktok.com/@openai/video/7623281227490970911",
        key="post_urls_text",
    )

    if st.button("Run post analysis", type="primary", use_container_width=True):
        post_urls = parse_lines(post_urls_text)

        if not post_urls:
            st.error("Please add at least one TikTok post URL.")
        elif missing:
            st.error("Please fill the missing configuration in the sidebar first.")
        else:
            with st.spinner("Fetching posts, downloading videos, analyzing content, and preparing payload..."):
                raw_df = load_analyze_tiktok_video(
                    gemini_api_key=gemini_api_key,
                    apify_api_token=apify_api_token,
                    postURLs=post_urls,
                    max_workers=max_workers,
                )

                if raw_df.empty:
                    st.warning("No videos were returned for these URLs.")
                else:
                    cleaned_df = clean_data(raw_df)
                    payload = prepare_payload(project_id, cleaned_df)
                    st.session_state["post_result"] = {
                        "raw_df": raw_df,
                        "cleaned_df": cleaned_df,
                        "payload": payload,
                    }

    render_results("post_result", api_url=api_url, api_headers=api_headers, project_id=project_id)

with search_tab:
    st.subheader("Analysis by Search")
    st.caption("Add one search query per line, then choose the date range and page size.")

    search_queries_text = st.text_area(
        "Search Queries",
        height=180,
        placeholder="openai\nchatgpt",
        key="search_queries_text",
    )

    search_col1, search_col2 = st.columns(2)
    with search_col1:
        date_range = st.selectbox("Date Range", options=list(DATE_RANGE_MAP.keys()), index=0)
    with search_col2:
        results_per_page = st.number_input("Results Per Page", min_value=1, max_value=100, value=10, step=1)

    if st.button("Run search analysis", use_container_width=True):
        search_queries = parse_lines(search_queries_text)

        if not search_queries:
            st.error("Please add at least one search query.")
        elif missing:
            st.error("Please fill the missing configuration in the sidebar first.")
        else:
            with st.spinner("Searching TikTok, downloading videos, analyzing content, and preparing payload..."):
                raw_df = load_analyze_tiktok_video(
                    gemini_api_key=gemini_api_key,
                    apify_api_token=apify_api_token,
                    searchQueries=search_queries,
                    date_range=date_range,
                    resultsPerPage=int(results_per_page),
                    max_workers=max_workers,
                )

                if raw_df.empty:
                    st.warning("No videos were returned for these search queries.")
                else:
                    cleaned_df = clean_data(raw_df)
                    payload = prepare_payload(project_id, cleaned_df)
                    st.session_state["search_result"] = {
                        "raw_df": raw_df,
                        "cleaned_df": cleaned_df,
                        "payload": payload,
                    }

    render_results("search_result", api_url=api_url, api_headers=api_headers, project_id=project_id)

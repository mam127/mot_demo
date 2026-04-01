import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import partial
from time import sleep
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qs, urlparse, urlunparse

import pandas as pd
import requests
from apify_client import ApifyClient
from google import genai
from pydantic import BaseModel, Field


DATE_RANGE_MAP = {
    "Past 24 hours": "1",
    "Past week": "2",
    "Past month": "3",
    "Last 3 months": "4",
    "Last 6 months": "5",
}


class VideoAnalysis(BaseModel):
    spoken_content_analysis: Optional[str] = Field(
        default=None,
        description="Analysis of the spoken/audio content in the video. Null if no spoken content is present.",
    )
    visual_content_analysis: str = Field(
        description="Analysis of visuals, on-screen text, editing style, music, and non-verbal cues."
    )
    audience: str = Field(description="Likely target audience of the video.")
    sentiment_analysis: str = Field(description="Overall emotional tone and sentiment of the video.")
    transcript: Optional[str] = Field(
        default=None,
        description="Transcript of spoken words only. Null if no spoken content is present.",
    )


# -----------------------------
# Core helpers
# -----------------------------

def run_parallel_calls(inputs, func, max_workers: int = 50):
    results = [None] * len(inputs)

    if not inputs:
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(func, item): i for i, item in enumerate(inputs)}

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                raise RuntimeError(f"Input at index {index} generated an exception: {exc}") from exc

    return results


def get_gemini_client(gemini_api_key: str):
    return genai.Client(api_key=gemini_api_key)


def get_apify_client(apify_api_token: str):
    return ApifyClient(apify_api_token)


# -----------------------------
# TikTok extraction
# -----------------------------

def get_tiktok_video(
    apify_api_token: str,
    postURLs: Optional[List[str]] = None,
    searchQueries: Optional[List[str]] = None,
    date_range: str = "Past 24 hours",
    resultsPerPage: int = 10,
) -> pd.DataFrame:
    postURLs = postURLs or []
    searchQueries = searchQueries or []

    if postURLs:
        run_input = {
            "commentsPerPost": 100,
            "excludePinnedPosts": False,
            "maxProfilesPerQuery": 1000,
            "maxRepliesPerComment": 10,
            "postURLs": postURLs,
            "proxyCountryCode": "None",
            "scrapeRelatedVideos": False,
            "shouldDownloadAvatars": False,
            "shouldDownloadCovers": False,
            "shouldDownloadMusicCovers": False,
            "shouldDownloadSlideshowImages": False,
            "shouldDownloadVideos": True,
        }
    elif searchQueries:
        run_input = {
            "commentsPerPost": 1000000000,
            "excludePinnedPosts": False,
            "resultsPerPage": resultsPerPage,
            "maxProfilesPerQuery": 100,
            "maxRepliesPerComment": 1000000000,
            "proxyCountryCode": "None",
            "scrapeRelatedVideos": False,
            "searchDatePosted": DATE_RANGE_MAP[date_range],
            "searchQueries": searchQueries,
            "searchSection": "/video",
            "searchSorting": "3",
            "shouldDownloadAvatars": False,
            "shouldDownloadCovers": False,
            "shouldDownloadMusicCovers": False,
            "shouldDownloadSlideshowImages": False,
            "shouldDownloadVideos": True,
        }
    else:
        raise ValueError("Either postURLs or searchQueries must be provided")

    apify_client = get_apify_client(apify_api_token)
    run = apify_client.actor("clockworks/tiktok-scraper").call(run_input=run_input)
    response = apify_client.dataset(run["defaultDatasetId"]).iterate_items()
    results = list(response)

    return pd.DataFrame(results) if results else pd.DataFrame()


# -----------------------------
# File / comment download helpers
# -----------------------------

def download_video(video_url: str, output_dir: str) -> str:
    filename = video_url.split("/")[-1].split("?")[0] or "video.mp4"
    output_file = os.path.join(output_dir, filename)

    with requests.get(video_url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(output_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    return output_file


def download_comments_data(full_url: str) -> List[Dict[str, Any]]:
    parsed = urlparse(full_url)
    url = urlunparse(parsed._replace(query=""))
    params = {k: v[0] for k, v in parse_qs(parsed.query).items()}

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


# -----------------------------
# Comment tree formatting
# -----------------------------

def parse_datetime(comment: Dict[str, Any]) -> datetime:
    create_time_iso = comment.get("createTimeISO")
    if create_time_iso:
        return datetime.fromisoformat(create_time_iso.replace("Z", "+00:00"))

    create_time = comment.get("createTime")
    if create_time is not None:
        return datetime.fromtimestamp(create_time, tz=timezone.utc)

    return datetime.min.replace(tzinfo=timezone.utc)


def normalize_date(comment: Dict[str, Any]) -> str:
    return parse_datetime(comment).date().isoformat()


def build_comment_tree(
    comments: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    nodes_by_cid: Dict[str, Dict[str, Any]] = {}

    for comment in comments:
        cid = comment.get("cid")
        if not cid:
            continue

        nodes_by_cid[cid] = {
            "cid": cid,
            "text": str(comment.get("text") or "").strip(),
            "date": normalize_date(comment),
            "dt": parse_datetime(comment),
            "username": str(comment.get("uniqueId") or "").strip(),
            "parent_cid": comment.get("repliesToId"),
            "replies": [],
        }

    roots: List[Dict[str, Any]] = []

    for node in nodes_by_cid.values():
        parent_cid = node["parent_cid"]
        if parent_cid and parent_cid in nodes_by_cid:
            nodes_by_cid[parent_cid]["replies"].append(node)
        else:
            roots.append(node)

    def sort_tree_desc(items: List[Dict[str, Any]]) -> None:
        items.sort(key=lambda x: x["dt"], reverse=True)
        for item in items:
            sort_tree_desc(item["replies"])

    sort_tree_desc(roots)
    return roots, nodes_by_cid



def format_comment_tree(tree: List[Dict[str, Any]], level: int = 0) -> str:
    lines: List[str] = []

    for node in tree:
        indent = "  " * level
        text = node["text"].replace("\n", " ").strip()
        lines.append(f"{indent}- {text} ({node['date']}, {node['username']})")

        if node["replies"]:
            child_text = format_comment_tree(node["replies"], level + 1)
            if child_text:
                lines.append(child_text)

    return "\n".join(lines)



def flatten_comments(tree: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []

    def walk(nodes: List[Dict[str, Any]]) -> None:
        for node in nodes:
            flat.append(node)
            if node["replies"]:
                walk(node["replies"])

    walk(tree)
    return flat



def get_latest_10_threaded(
    tree: List[Dict[str, Any]],
    nodes_by_cid: Dict[str, Dict[str, Any]],
    limit: int = 10,
) -> str:
    flat_comments = flatten_comments(tree)
    latest = sorted(flat_comments, key=lambda x: x["dt"], reverse=True)[:limit]

    selected_cids: Set[str] = {node["cid"] for node in latest}
    cids_to_keep: Set[str] = set(selected_cids)

    for node in latest:
        parent_cid = node.get("parent_cid")
        while parent_cid and parent_cid in nodes_by_cid:
            cids_to_keep.add(parent_cid)
            parent_cid = nodes_by_cid[parent_cid].get("parent_cid")

    def prune_tree(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pruned: List[Dict[str, Any]] = []

        for node in nodes:
            pruned_replies = prune_tree(node["replies"])
            if node["cid"] in cids_to_keep or pruned_replies:
                pruned.append(
                    {
                        "cid": node["cid"],
                        "text": node["text"],
                        "date": node["date"],
                        "dt": node["dt"],
                        "username": node["username"],
                        "parent_cid": node["parent_cid"],
                        "replies": pruned_replies,
                    }
                )

        pruned.sort(key=lambda x: x["dt"], reverse=True)
        return pruned

    pruned_tree = prune_tree(tree)
    return format_comment_tree(pruned_tree)



def process_comments(comments: List[Dict[str, Any]], latest_limit: int = 10) -> Tuple[str, str]:
    roots, nodes_by_cid = build_comment_tree(comments)
    full_comments_text = format_comment_tree(roots)
    latest_10_text = get_latest_10_threaded(roots, nodes_by_cid, limit=latest_limit)
    return full_comments_text, latest_10_text


# -----------------------------
# Gemini analysis
# -----------------------------

def analyze_tiktok_video(
    filename: str,
    gemini_api_key: str,
    model: str = "gemini-3-flash-preview",
    poll_interval_seconds: int = 5,
) -> Dict[str, Any]:
    prompt = """
You are analyzing a TikTok video using all available inputs:
spoken audio, on-screen text, visuals, music/sound effects, and editing style.

Your job is to understand the video as completely as possible and return a structured analysis.

Instructions:
- Analyze the video using these fields:
    1. spoken_content_analysis
    2. visual_content_analysis
    3. audience
    4. sentiment_analysis
    5. transcript
- If no spoken content is present:
    - spoken_content_analysis must be null
    - transcript must be null
- The output should be in English, except for the transcript which should be in the language of the video.
- Return only valid JSON matching the schema.
"""

    gemini_client = get_gemini_client(gemini_api_key)
    uploaded_file = gemini_client.files.upload(file=filename)

    while not uploaded_file.state or uploaded_file.state.name != "ACTIVE":
        sleep(poll_interval_seconds)
        uploaded_file = gemini_client.files.get(name=uploaded_file.name)

    response = gemini_client.models.generate_content(
        model=model,
        contents=[uploaded_file, prompt.strip()],
        config={
            "response_mime_type": "application/json",
            "response_json_schema": VideoAnalysis.model_json_schema(),
        },
    )

    return VideoAnalysis.model_validate_json(response.text).model_dump()



def generate_tiktok_comment_summary(
    context: str,
    comments: str,
    gemini_api_key: str,
    model: str = "gemini-3-flash-preview",
) -> str:
    prompt = f"""Your task is to summarize the comments of a TikTok video.

Here is the context of the video:
{context}

Summarize the comments of the video in English.
Return only the summary text.

Comments:
{comments}
"""

    gemini_client = get_gemini_client(gemini_api_key)
    response = gemini_client.models.generate_content(model=model, contents=prompt)
    return response.text.strip()



def process_comments_fully(row: Dict[str, Any], gemini_api_key: str):
    if row.get("commentCount", 0) == 0:
        return None, None

    comments_data = download_comments_data(row["commentsDatasetUrl"])
    full_comments_text, latest_10_text = process_comments(comments_data)

    analysis_context = row.get("analysis")
    if isinstance(analysis_context, dict):
        analysis_context = json.dumps(analysis_context, ensure_ascii=False)

    comment_summary = generate_tiktok_comment_summary(
        analysis_context or "",
        full_comments_text[:2500000],
        gemini_api_key=gemini_api_key,
    )

    return latest_10_text, comment_summary


# -----------------------------
# Output shaping
# -----------------------------

def create_message(row: pd.Series) -> str:
    message = ""

    if row.get("text"):
        message += f"**Caption:** {row['text']}\n"
    if row.get("spoken_content_analysis"):
        message += f"**Spoken Content Analysis:** {row['spoken_content_analysis']}\n"
    if row.get("visual_content_analysis"):
        message += f"**Visual Analysis:** {row['visual_content_analysis']}\n"
    if row.get("audience"):
        message += f"**Audience:** {row['audience']}\n"
    if row.get("sentiment_analysis"):
        message += f"**Sentiment Analysis:** {row['sentiment_analysis']}\n"
    if row.get("transcript"):
        message += f"**Transcript:** {row['transcript']}\n"
    if row.get("comments_summary"):
        message += f"**Comments Summary:** {row['comments_summary']}\n"
    if row.get("comments_txt"):
        message += f"**Latest Comments:**\n{row['comments_txt']}\n"

    message = message.replace("\n", "\n\n")
    message = re.sub("\n{2,}", "\n\n", message)
    return message



def format_iso_zulu(date_str: str) -> str:
    if "." in date_str:
        dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    else:
        dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")



def is_effectively_null(val) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and pd.isna(val):
        return True
    if hasattr(val, "size") and getattr(val, "size", None) == 0:
        return True
    if isinstance(val, (list, tuple, dict, set)) and len(val) == 0:
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    return False



def df_to_batch(df: pd.DataFrame) -> List[Dict[str, Any]]:
    batch = []
    for _, row in df.iterrows():
        entry: Dict[str, Any] = {}
        filters: Dict[str, Any] = {}
        attributes: Dict[str, Any] = {}

        for col in df.columns:
            val = row[col]
            if is_effectively_null(val):
                continue

            if col.endswith("__filter"):
                filters[col.replace("__filter", "")] = val
            elif col.endswith("__attribute"):
                attributes[col.replace("__attribute", "")] = val
            else:
                entry[col] = val

        if filters:
            entry["filters"] = filters
        if attributes:
            entry["attributes"] = attributes

        batch.append(entry)

    return batch



def prepare_payload(project_id: str, df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "project_id": project_id,
        "taxonomy_id": "default",
        "skip_steps": {},
        "batch": df_to_batch(df),
    }



def send_to_api(payload: Dict[str, Any], api_url: str, api_headers: Dict[str, str]):
    if not api_url.strip():
        return False, None, "API_URL is not set. (Sending is disabled.)"

    try:
        resp = requests.post(api_url.strip(), json=payload, headers=api_headers, timeout=30)
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        return resp.ok, resp.status_code, body
    except requests.exceptions.RequestException as e:
        return False, None, str(e)


def normalize_filter_list(value):
    if value is None:
        return None

    if isinstance(value, float) and pd.isna(value):
        return None

    if not isinstance(value, list):
        if isinstance(value, (str, int, float, bool)):
            return [value]
        return [str(value)]

    cleaned = []
    for item in value:
        if item is None:
            continue

        if isinstance(item, (str, int, float, bool)):
            cleaned.append(item)
            continue

        if isinstance(item, dict):
            for key in ["name", "hashtagName", "tagName", "title", "text", "keyword", "uniqueId"]:
                if key in item and item[key] not in [None, ""]:
                    cleaned.append(str(item[key]))
                    break
            continue

        cleaned.append(str(item))

    return cleaned or None
    

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "hashtags" in df.columns:
        df["hashtags"] = df["hashtags"].apply(normalize_filter_list)

    if "mentions" in df.columns:
        df["mentions"] = df["mentions"].apply(normalize_filter_list)
    
    df["message"] = df.apply(create_message, axis=1)
    df["original_message"] = df["message"]

    authors_df = pd.DataFrame(df["authorMeta"].tolist())
    authors_df.columns = ["user_" + col for col in authors_df.columns]
    df = pd.concat([df, authors_df], axis=1)

    df["TikTok Videos: video duration__filter"] = df["videoMeta"].apply(
        lambda x: x.get("duration") if isinstance(x, dict) else None
    )

    df = df.drop(
        columns=[
            "createTime",
            "authorMeta",
            "musicMeta",
            "videoMeta",
            "detailedMentions",
            "effectStickers",
            "isSlideshow",
            "mediaUrls",
            "submittedVideoUrl",
            "commentsDatasetUrl",
            "download_url",
            "download_path",
            "spoken_content_analysis",
            "visual_content_analysis",
            "audience",
            "sentiment_analysis",
            "transcript",
            "analysis",
            "comments_txt",
            "comments_summary",
            "user_profileUrl",
            "user_nickName",
            "user_avatar",
            "user_verified",
            "user_signature",
            "user_roomId",
            "user_ttSeller",
            "user_createTime",
            "user_originalAvatarUrl",
            "isMuted"
        ],
        errors="ignore",
    )

    df = df.rename(
        columns={
            "id": "ticket_id",
            "text": "TikTok Videos: caption__attribute",
            "textLanguage": "TikTok Videos: caption language__filter",
            "createTimeISO": "ds",
            "locationCreated": "TikTok Videos: location__filter",
            "isAd": "TikTok Videos: is ad__filter",
            "webVideoUrl": "custom_url",
            "diggCount": "TikTok Videos: likes count__filter",
            "shareCount": "TikTok Videos: shares count__filter",
            "playCount": "TikTok Videos: views count__filter",
            "collectCount": "TikTok Videos: saves count__filter",
            "commentCount": "TikTok Videos: comments count__filter",
            "repostCount": "TikTok Videos: reposts count__filter",
            "mentions": "TikTok Videos: mentions__filter",
            "hashtags": "TikTok Videos: hashtags__filter",
            "isPinned": "TikTok Videos: is pinned__filter",
            "isSponsored": "TikTok Videos: is sponsored__filter",
            "user_privateAccount": "TikTok Videos: user private account__filter",
            "user_following": "TikTok Videos: user following__filter",
            "user_friends": "TikTok Videos: user friends__filter",
            "user_fans": "TikTok Videos: user fans__filter",
            "user_heart": "TikTok Videos: user heart__filter",
            "user_video": "TikTok Videos: user video__filter",
            "user_digg": "TikTok Videos: user digg__filter",
        }
    )

    df["ds"] = df["ds"].apply(format_iso_zulu)
    df["source"] = "Tiktok"

    final_cols = ['ticket_id', 'TikTok Videos: caption__attribute',
       'TikTok Videos: caption language__filter', 'ds',
       'TikTok Videos: location__filter', 'TikTok Videos: is ad__filter',
       'custom_url', 'TikTok Videos: likes count__filter',
       'TikTok Videos: shares count__filter',
       'TikTok Videos: views count__filter',
       'TikTok Videos: saves count__filter',
       'TikTok Videos: comments count__filter',
       'TikTok Videos: reposts count__filter',
       'TikTok Videos: mentions__filter', 'TikTok Videos: hashtags__filter',
       'TikTok Videos: is pinned__filter',
       'TikTok Videos: is sponsored__filter', 'message', 'original_message',
       'user_id', 'user_name', 'TikTok Videos: user private account__filter',
       'TikTok Videos: user following__filter',
       'TikTok Videos: user friends__filter',
       'TikTok Videos: user fans__filter', 'TikTok Videos: user heart__filter',
       'TikTok Videos: user video__filter', 'TikTok Videos: user digg__filter',
       'TikTok Videos: video duration__filter', 'source']

    cols_to_drop = [c for c in df.columns if c not in final_cols]
    df = df.drop(cols_to_drop, axis=1, errors='ignore')
    
    return df



def load_analyze_tiktok_video(
    gemini_api_key: str,
    apify_api_token: str,
    postURLs: Optional[List[str]] = None,
    searchQueries: Optional[List[str]] = None,
    date_range: str = "Past 24 hours",
    resultsPerPage: int = 10,
    max_workers: int = 4,
) -> pd.DataFrame:
    df = get_tiktok_video(
        apify_api_token=apify_api_token,
        postURLs=postURLs,
        searchQueries=searchQueries,
        date_range=date_range,
        resultsPerPage=resultsPerPage,
    )

    if df.empty:
        return df

    with tempfile.TemporaryDirectory() as temp_dir:
        df = df.copy()
        df["download_url"] = df["videoMeta"].apply(lambda x: x.get("downloadAddr") if isinstance(x, dict) else None)

        download_func = partial(download_video, output_dir=temp_dir)
        df["download_path"] = run_parallel_calls(df["download_url"].tolist(), download_func, max_workers=max_workers)

        analysis_func = partial(analyze_tiktok_video, gemini_api_key=gemini_api_key)
        analysis = run_parallel_calls(df["download_path"].tolist(), analysis_func, max_workers=max_workers)
        df["analysis"] = analysis

        analysis_df = pd.DataFrame(analysis)
        df = pd.concat([df, analysis_df], axis=1)

        comments_func = partial(process_comments_fully, gemini_api_key=gemini_api_key)
        comments_processed = run_parallel_calls(df.to_dict("records"), comments_func, max_workers=max_workers)
        df[["comments_txt", "comments_summary"]] = pd.DataFrame(comments_processed, index=df.index)

    return df



def parse_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]

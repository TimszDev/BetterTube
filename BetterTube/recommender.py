from __future__ import annotations

import json
import re
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from collections import Counter
from typing import Any, Iterable

from yt_dlp import YoutubeDL

from db import (
    add_watched_video,
    get_recommendation_feedback_entries,
    get_watched_video_ids,
    get_watched_videos,
    log_recommendations,
)

TOKEN_RE = re.compile(r"[a-z0-9]+")
URL_RE = re.compile(r"https?://[^\s<>\"]+")
URL_STRIP_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
STOPWORDS = {
    "about",
    "after",
    "again",
    "all",
    "also",
    "and",
    "are",
    "been",
    "before",
    "being",
    "between",
    "both",
    "but",
    "can",
    "could",
    "did",
    "does",
    "doing",
    "done",
    "for",
    "from",
    "get",
    "got",
    "has",
    "have",
    "how",
    "his",
    "him",
    "her",
    "hers",
    "she",
    "he",
    "had",
    "into",
    "its",
    "just",
    "know",
    "known",
    "like",
    "make",
    "more",
    "most",
    "much",
    "new",
    "not",
    "now",
    "off",
    "one",
    "our",
    "out",
    "over",
    "really",
    "see",
    "some",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "through",
    "too",
    "under",
    "very",
    "want",
    "was",
    "were",
    "what",
    "would",
    "when",
    "where",
    "which",
    "while",
    "who",
    "why",
    "will",
    "with",
    "you",
    "your",
    "http",
    "https",
    "www",
    "com",
    "youtube",
    "youtu",
    "watch",
    "channel",
    "video",
    "captions",
    "chapters",
    "chapter",
    "year",
    "years",
    "time",
    "times",
    "full",
    "official",
    "episode",
    "part",
}
CAPTION_LANG_PRIORITY = ("en", "en-us", "en-gb", "a.en")
CAPTION_EXT_PRIORITY = ("json3", "srv3", "ttml", "vtt", "srv2", "srv1")


class _YTDLLogger:
    def debug(self, msg: str) -> None:
        return None

    def warning(self, msg: str) -> None:
        return None

    def error(self, msg: str) -> None:
        return None


def _is_youtube_host(host: str) -> bool:
    clean_host = host.lower().strip()
    if clean_host.startswith("www."):
        clean_host = clean_host[4:]
    return clean_host == "youtu.be" or clean_host == "youtube.com" or clean_host.endswith(".youtube.com")


def extract_youtube_links(text: str) -> list[str]:
    links: list[str] = []
    seen: set[str] = set()
    for match in URL_RE.findall(text or ""):
        url = match.strip().rstrip("),.;]}>,")
        if not url:
            continue
        parsed = urlparse(url)
        if not _is_youtube_host(parsed.netloc):
            continue
        dedupe_key = url.lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        links.append(url)
    return links


def _tokenize(text: str) -> list[str]:
    cleaned = URL_STRIP_RE.sub(" ", text or "")
    tokens = TOKEN_RE.findall(cleaned.lower())
    return [token for token in tokens if len(token) > 2 and token not in STOPWORDS and not token.isdigit()]


def _normalize_words(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        word = str(value or "").strip()
        if not word:
            continue
        output.append(word)
    return output


def _format_duration(seconds: int) -> str:
    seconds = max(0, int(seconds or 0))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{sec:02d}"
    return f"{minutes}:{sec:02d}"


def _extract_info(url: str) -> dict[str, Any]:
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "logger": _YTDLLogger(),
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if not info:
        raise ValueError("No metadata returned by YouTube.")
    if isinstance(info, dict) and "entries" in info and info["entries"]:
        info = info["entries"][0]
    if not isinstance(info, dict):
        raise ValueError("Unexpected metadata format from YouTube.")
    return info


def _download_text(url: str, max_bytes: int = 800_000) -> str:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=12) as response:
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        data = data[:max_bytes]
    return data.decode("utf-8", errors="ignore")


def _caption_plain_text_from_json3(raw: str) -> str:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ""

    events = payload.get("events") or []
    chunks: list[str] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        segments = event.get("segs") or []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            value = str(segment.get("utf8") or "").strip()
            if value:
                chunks.append(value)
    return " ".join(chunks)


def _caption_plain_text_from_markup(raw: str) -> str:
    text = raw
    text = re.sub(r"(?im)^\s*WEBVTT.*$", " ", text)
    text = re.sub(r"(?m)^\s*\d+\s*$", " ", text)
    text = re.sub(r"(?m)^\s*\d{1,2}:\d{2}(?::\d{2})?[.,]\d+\s+-->\s+\d{1,2}:\d{2}(?::\d{2})?[.,]\d+.*$", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\{\\an\d+\}", " ", text)
    text = re.sub(r"&[a-zA-Z#0-9]+;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_caption_excerpt(info: dict[str, Any], max_chars: int = 5000) -> str:
    candidates: list[tuple[int, str, str]] = []
    for source_name in ("subtitles", "automatic_captions"):
        source = info.get(source_name) or {}
        if not isinstance(source, dict):
            continue
        for language, tracks in source.items():
            if not isinstance(tracks, list):
                continue
            language_value = str(language or "").lower()
            try:
                lang_rank = CAPTION_LANG_PRIORITY.index(language_value)
            except ValueError:
                lang_rank = len(CAPTION_LANG_PRIORITY) + 1

            for track in tracks:
                if not isinstance(track, dict):
                    continue
                track_url = str(track.get("url") or "").strip()
                if not track_url:
                    continue
                ext_value = str(track.get("ext") or "").lower()
                try:
                    ext_rank = CAPTION_EXT_PRIORITY.index(ext_value)
                except ValueError:
                    ext_rank = len(CAPTION_EXT_PRIORITY) + 1
                score = (lang_rank * 100) + ext_rank
                candidates.append((score, track_url, ext_value))

    candidates.sort(key=lambda item: item[0])
    for _, caption_url, ext_value in candidates[:6]:
        try:
            raw_text = _download_text(caption_url)
        except Exception:
            continue

        if ext_value == "json3":
            plain_text = _caption_plain_text_from_json3(raw_text)
        else:
            plain_text = _caption_plain_text_from_markup(raw_text)
        if not plain_text:
            continue
        return plain_text[:max_chars]

    return ""


def _extract_subjects(info: dict[str, Any], tags: list[str]) -> list[str]:
    categories = _normalize_words(info.get("categories") or [])
    if categories:
        return categories[:6]

    text = " ".join(
        [
            info.get("title") or "",
            info.get("description") or "",
            " ".join(tags),
        ]
    )
    tokens = _tokenize(text)
    if not tokens:
        return []
    counts = Counter(tokens)
    return [term for term, _ in counts.most_common(6)]


def extract_video_metadata(url: str) -> dict[str, Any]:
    info = _extract_info(url)

    video_id = str(info.get("id") or "").strip()
    if not video_id:
        raise ValueError("Could not find a YouTube video id for this link.")

    title = str(info.get("title") or "").strip()
    if not title:
        raise ValueError("Could not read the video title.")

    creator = str(info.get("channel") or info.get("uploader") or "Unknown creator").strip()
    description = info.get("description") or ""
    duration_seconds = int(info.get("duration") or 0)
    chapters = info.get("chapters") or []

    tags = _normalize_words(info.get("tags") or [])
    subjects = _extract_subjects(info, tags)
    chapter_titles = [
        str(chapter.get("title") or "").strip()
        for chapter in chapters
        if isinstance(chapter, dict) and str(chapter.get("title") or "").strip()
    ]
    caption_excerpt = _extract_caption_excerpt(info)
    context_parts: list[str] = []
    if chapter_titles:
        context_parts.append("Chapters: " + " | ".join(chapter_titles[:12]))
    if caption_excerpt:
        context_parts.append("Captions: " + caption_excerpt)
    if not context_parts and description:
        context_parts.append(description[:1200])
    context_text = " ".join(context_parts).strip()[:8000]

    return {
        "video_id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "title": title,
        "creator": creator,
        "duration_seconds": duration_seconds,
        "duration_text": _format_duration(duration_seconds),
        "description": description,
        "context_text": context_text,
        "tags": tags[:40],
        "subjects": subjects,
    }


def _unique_limited(tokens: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= limit:
            break
    return out


def _build_profile(watched_videos: list[dict[str, Any]]) -> dict[str, Any]:
    term_weights: Counter[str] = Counter()
    channel_counts: Counter[str] = Counter()
    durations: list[int] = []

    for item in watched_videos:
        creator = (item.get("creator") or "").strip().lower()
        if creator:
            channel_counts[creator] += 1

        duration = int(item.get("duration_seconds") or 0)
        if duration > 0:
            durations.append(duration)

        tag_tokens: set[str] = set()
        for tag in item.get("tags") or []:
            tag_tokens.update(_tokenize(tag))
        for token in tag_tokens:
            term_weights[token] += 3

        subject_tokens: set[str] = set()
        for subject in item.get("subjects") or []:
            subject_tokens.update(_tokenize(subject))
        for token in subject_tokens:
            term_weights[token] += 2

        for token in set(_tokenize(item.get("title") or "")):
            term_weights[token] += 2

        desc_tokens = _unique_limited(_tokenize(item.get("description") or ""), 60)
        for token in desc_tokens:
            term_weights[token] += 1

        context_tokens = _unique_limited(_tokenize(item.get("context_text") or ""), 140)
        for token in context_tokens:
            term_weights[token] += 2

    top_terms = [term for term, _ in term_weights.most_common(15)]
    keyword_normalizer = sum(weight for _, weight in term_weights.most_common(20)) or 1
    avg_duration = int(sum(durations) / len(durations)) if durations else 0
    max_channel_count = max(channel_counts.values()) if channel_counts else 1

    return {
        "term_weights": term_weights,
        "top_terms": top_terms,
        "channel_counts": channel_counts,
        "max_channel_count": max_channel_count,
        "avg_duration": avg_duration,
        "keyword_normalizer": keyword_normalizer,
    }


def _build_search_queries(profile: dict[str, Any]) -> list[str]:
    top_terms = profile["top_terms"]
    channel_counts = profile["channel_counts"]

    queries: list[str] = []
    if top_terms:
        queries.append(" ".join(top_terms[:3]))
    if len(top_terms) >= 6:
        queries.append(" ".join(top_terms[3:6]))
    if len(top_terms) >= 9:
        queries.append(" ".join(top_terms[6:9]))

    for channel_name, channel_count in channel_counts.most_common(2):
        if channel_count < 3:
            continue
        if top_terms:
            queries.append(f"{' '.join(top_terms[:2])} {channel_name}")
        else:
            queries.append(channel_name)

    cleaned: list[str] = []
    seen: set[str] = set()
    for query in queries:
        text = query.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned[:5]


def _search_candidates(query: str, max_results: int = 12) -> list[dict[str, Any]]:
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": "in_playlist",
        "ignoreerrors": True,
        "logger": _YTDLLogger(),
    }
    with YoutubeDL(ydl_opts) as ydl:
        results = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)

    entries = (results or {}).get("entries") or []
    output: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        video_id = str(entry.get("id") or "").strip()
        if not video_id:
            continue
        title = str(entry.get("title") or "").strip()
        creator = str(entry.get("channel") or entry.get("uploader") or "Unknown creator").strip()
        duration_seconds = int(entry.get("duration") or 0)
        output.append(
            {
                "video_id": video_id,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "title": title or video_id,
                "creator": creator,
                "duration_seconds": duration_seconds,
                "duration_text": _format_duration(duration_seconds),
                "description": "",
                "context_text": "",
                "tags": [],
                "subjects": [],
            }
        )
    return output


def _build_feedback_profile(feedback_entries: list[dict[str, Any]]) -> dict[str, Any]:
    liked_terms: Counter[str] = Counter()
    disliked_terms: Counter[str] = Counter()
    liked_creators: Counter[str] = Counter()
    disliked_creators: Counter[str] = Counter()

    for entry in feedback_entries:
        feedback_value = int(entry.get("feedback") or 0)
        if feedback_value not in {-1, 1}:
            continue

        creator = (entry.get("creator") or "").strip().lower()
        if creator:
            if feedback_value > 0:
                liked_creators[creator] += 1
            else:
                disliked_creators[creator] += 1

        token_pool = set(_tokenize(entry.get("title") or ""))
        token_pool.update(_tokenize(entry.get("creator") or ""))
        token_pool.update(_unique_limited(_tokenize(entry.get("description") or ""), 35))
        token_pool.update(_unique_limited(_tokenize(entry.get("context_text") or ""), 50))
        for tag in entry.get("tags") or []:
            token_pool.update(_tokenize(tag))
        for subject in entry.get("subjects") or []:
            token_pool.update(_tokenize(subject))
        for token in token_pool:
            if feedback_value > 0:
                liked_terms[token] += 2
            else:
                disliked_terms[token] += 2

    term_norm = sum(v for _, v in liked_terms.most_common(15)) + sum(v for _, v in disliked_terms.most_common(15))
    creator_norm = max(1, (max(liked_creators.values()) if liked_creators else 0), (max(disliked_creators.values()) if disliked_creators else 0))

    return {
        "liked_terms": liked_terms,
        "disliked_terms": disliked_terms,
        "liked_creators": liked_creators,
        "disliked_creators": disliked_creators,
        "term_norm": max(1, term_norm),
        "creator_norm": creator_norm,
    }


def _candidate_token_set(candidate: dict[str, Any]) -> set[str]:
    tokens = set()
    tokens.update(_tokenize(candidate.get("title") or ""))
    tokens.update(_tokenize(candidate.get("description") or ""))
    tokens.update(_tokenize(candidate.get("context_text") or ""))
    for tag in candidate.get("tags") or []:
        tokens.update(_tokenize(tag))
    for subject in candidate.get("subjects") or []:
        tokens.update(_tokenize(subject))
    return tokens


def _is_dislike_dominant(candidate: dict[str, Any], feedback_profile: dict[str, Any]) -> bool:
    liked_terms: Counter[str] = feedback_profile["liked_terms"]
    disliked_terms: Counter[str] = feedback_profile["disliked_terms"]
    liked_creators: Counter[str] = feedback_profile["liked_creators"]
    disliked_creators: Counter[str] = feedback_profile["disliked_creators"]

    candidate_tokens = _candidate_token_set(candidate)
    liked_term_weight = sum(liked_terms[token] for token in candidate_tokens if token in liked_terms)
    disliked_term_weight = sum(disliked_terms[token] for token in candidate_tokens if token in disliked_terms)

    creator = (candidate.get("creator") or "").strip().lower()
    liked_creator_hits = liked_creators.get(creator, 0)
    disliked_creator_hits = disliked_creators.get(creator, 0)

    if disliked_creator_hits >= liked_creator_hits + 3 and disliked_term_weight >= liked_term_weight + 10:
        return True
    if disliked_term_weight >= liked_term_weight + 20 and disliked_term_weight >= 20:
        return True
    return False


def _score_candidate(
    candidate: dict[str, Any],
    profile: dict[str, Any],
    feedback_profile: dict[str, Any] | None = None,
) -> tuple[float, list[str]]:
    profile_terms: Counter[str] = profile["term_weights"]
    profile_channels: Counter[str] = profile["channel_counts"]

    candidate_tokens = _candidate_token_set(candidate)

    overlap = [token for token in candidate_tokens if token in profile_terms]
    overlap.sort(key=lambda token: profile_terms[token], reverse=True)

    keyword_raw_score = sum(profile_terms[token] for token in overlap)
    keyword_score = min(1.0, keyword_raw_score / profile["keyword_normalizer"])

    creator = (candidate.get("creator") or "").strip().lower()
    creator_count = profile_channels.get(creator, 0)
    creator_score = creator_count / profile["max_channel_count"] if creator_count else 0.0

    avg_duration = profile.get("avg_duration") or 0
    cand_duration = int(candidate.get("duration_seconds") or 0)
    if avg_duration > 0 and cand_duration > 0:
        duration_delta = abs(cand_duration - avg_duration) / max(avg_duration, 1)
        duration_score = max(0.0, 1.0 - min(duration_delta, 1.0))
    else:
        duration_score = 0.2

    score = (0.75 * keyword_score) + (0.15 * creator_score) + (0.10 * duration_score)

    if feedback_profile:
        liked_terms: Counter[str] = feedback_profile["liked_terms"]
        disliked_terms: Counter[str] = feedback_profile["disliked_terms"]
        liked_creators: Counter[str] = feedback_profile["liked_creators"]
        disliked_creators: Counter[str] = feedback_profile["disliked_creators"]

        liked_term_weight = sum(liked_terms[token] for token in candidate_tokens if token in liked_terms)
        disliked_term_weight = sum(disliked_terms[token] for token in candidate_tokens if token in disliked_terms)
        feedback_term_raw = liked_term_weight - (1.45 * disliked_term_weight)
        feedback_term_score = max(-1.0, min(1.0, feedback_term_raw / feedback_profile["term_norm"]))

        creator_feedback_raw = liked_creators.get(creator, 0) - (1.7 * disliked_creators.get(creator, 0))
        feedback_creator_score = max(-1.0, min(1.0, creator_feedback_raw / feedback_profile["creator_norm"]))

        score += (0.14 * feedback_term_score) + (0.09 * feedback_creator_score)

        if disliked_term_weight >= liked_term_weight + 12 and liked_term_weight == 0:
            score -= 0.12
        if disliked_creators.get(creator, 0) > liked_creators.get(creator, 0):
            score -= 0.09

    # small boost for videos that match at least one strong profile term
    if overlap:
        score += min(0.08, profile_terms[overlap[0]] / max(profile["keyword_normalizer"], 1))

    return max(0.0, min(score, 1.0)), overlap[:4]


def train_video(url: str) -> tuple[bool, dict[str, Any]]:
    metadata = extract_video_metadata(url)
    added = add_watched_video(metadata)
    return added, metadata


def train_video_from_suggestion(
    video_id: str,
    url: str,
    title: str,
    creator: str,
    duration_seconds: int = 0,
    metadata: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any]]:
    if metadata:
        normalized = dict(metadata)
        if not normalized.get("context_text"):
            normalized["context_text"] = ""
        added = add_watched_video(normalized)
        return added, normalized

    try:
        return train_video(url)
    except Exception:
        fallback = {
            "video_id": str(video_id or "").strip(),
            "url": str(url or "").strip(),
            "title": str(title or "").strip() or str(video_id or "Unknown video"),
            "creator": str(creator or "").strip() or "Unknown creator",
            "duration_seconds": int(duration_seconds or 0),
            "duration_text": _format_duration(int(duration_seconds or 0)),
            "description": "",
            "context_text": "",
            "tags": [],
            "subjects": [],
        }
        if not fallback["video_id"] or not fallback["url"]:
            raise ValueError("Could not add liked suggestion to training history.")
        added = add_watched_video(fallback)
        return added, fallback


def train_videos_from_text(text: str) -> dict[str, Any]:
    links = extract_youtube_links(text)
    added_count = 0
    already_known_count = 0
    failed: list[dict[str, str]] = []

    for link in links:
        try:
            added, _ = train_video(link)
        except Exception as exc:
            failed.append({"url": link, "error": str(exc)})
            continue

        if added:
            added_count += 1
        else:
            already_known_count += 1

    return {
        "total_links": len(links),
        "added_count": added_count,
        "already_known_count": already_known_count,
        "failed_count": len(failed),
        "failed": failed,
    }


def suggest_videos(
    count: int = 3,
    exclude_video_ids: set[str] | None = None,
    log_results: bool = True,
) -> list[dict[str, Any]]:
    watched_videos = get_watched_videos()
    if len(watched_videos) < 1:
        return []

    profile = _build_profile(watched_videos)
    feedback_entries = get_recommendation_feedback_entries()
    feedback_profile = _build_feedback_profile(feedback_entries)
    feedback_by_id = {str(entry.get("video_id") or ""): int(entry.get("feedback") or 0) for entry in feedback_entries}
    excluded_ids = {str(item).strip() for item in (exclude_video_ids or set()) if str(item).strip()}
    queries = _build_search_queries(profile)
    if not queries:
        return []

    watched_ids = get_watched_video_ids()
    candidate_pool: dict[str, dict[str, Any]] = {}
    for query in queries:
        for candidate in _search_candidates(query):
            candidate_id = candidate["video_id"]
            if candidate_id in watched_ids:
                continue
            if candidate_id in excluded_ids:
                continue
            if feedback_by_id.get(candidate_id) in {-1, 1}:
                continue
            prelim_score, _ = _score_candidate(candidate, profile, feedback_profile)
            existing = candidate_pool.get(candidate_id)
            if existing and existing["prelim_score"] >= prelim_score:
                continue
            candidate["prelim_score"] = prelim_score
            candidate_pool[candidate_id] = candidate
            if len(candidate_pool) >= 45:
                break
        if len(candidate_pool) >= 45:
            break

    if not candidate_pool:
        return []

    finalists = sorted(candidate_pool.values(), key=lambda item: item["prelim_score"], reverse=True)
    finalists = finalists[: max(4, count * 2)]
    detailed_limit = max(3, count)

    candidates: list[dict[str, Any]] = []
    for index, lightweight in enumerate(finalists):
        base_item = lightweight
        if index < detailed_limit:
            try:
                metadata = extract_video_metadata(lightweight["url"])
                base_item = metadata
            except Exception:
                pass

        if base_item["video_id"] in watched_ids:
            continue
        if base_item["video_id"] in excluded_ids:
            continue
        if feedback_by_id.get(base_item["video_id"]) in {-1, 1}:
            continue
        if _is_dislike_dominant(base_item, feedback_profile):
            continue

        score, matched_terms = _score_candidate(base_item, profile, feedback_profile)
        if score < 0.08:
            continue

        base_item["score"] = round(score, 3)
        base_item["matched_terms"] = matched_terms
        candidates.append(base_item)

    if not candidates:
        # Relaxed fallback so suggestions still appear when negative feedback is very broad.
        for lightweight in finalists:
            base_item = lightweight
            if base_item["video_id"] in watched_ids:
                continue
            if base_item["video_id"] in excluded_ids:
                continue
            if feedback_by_id.get(base_item["video_id"]) in {-1, 1}:
                continue

            score, matched_terms = _score_candidate(base_item, profile, feedback_profile)
            if score <= 0.0:
                continue
            base_item["score"] = round(score, 3)
            base_item["matched_terms"] = matched_terms
            candidates.append(base_item)

    if not candidates:
        # Last resort: ignore feedback shaping but keep hard exclusions.
        for lightweight in finalists:
            base_item = lightweight
            if base_item["video_id"] in watched_ids:
                continue
            if base_item["video_id"] in excluded_ids:
                continue
            if feedback_by_id.get(base_item["video_id"]) in {-1, 1}:
                continue

            score, matched_terms = _score_candidate(base_item, profile, None)
            if score <= 0.0:
                continue
            base_item["score"] = round(score, 3)
            base_item["matched_terms"] = matched_terms
            candidates.append(base_item)

    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = []
    seen_ids: set[str] = set()
    creator_counts: Counter[str] = Counter()
    overflow: list[dict[str, Any]] = []
    for item in candidates:
        if item["video_id"] in seen_ids:
            continue
        creator_key = (item.get("creator") or "").strip().lower()
        liked_creator_count = feedback_profile["liked_creators"].get(creator_key, 0)
        creator_cap = 2 if liked_creator_count > 0 else 1
        if creator_counts[creator_key] >= creator_cap:
            overflow.append(item)
            continue

        seen_ids.add(item["video_id"])
        creator_counts[creator_key] += 1
        selected.append(item)
        if len(selected) >= count:
            break

    if len(selected) < count:
        for item in overflow:
            if item["video_id"] in seen_ids:
                continue
            seen_ids.add(item["video_id"])
            selected.append(item)
            if len(selected) >= count:
                break

    if log_results:
        log_recommendations(selected)
    return selected

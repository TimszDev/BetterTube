from __future__ import annotations

from typing import Any

from flask import Flask, flash, redirect, render_template, request, url_for

from db import (
    clear_recommendations,
    get_active_suggestions,
    get_recommendation_feedback_map,
    get_video_count,
    init_db,
    list_trained_videos,
    set_active_suggestions,
    set_recommendation_feedback,
)
from recommender import (
    extract_video_metadata,
    suggest_videos,
    train_video,
    train_video_from_suggestion,
    train_videos_from_text,
)

app = Flask(__name__)
app.config["SECRET_KEY"] = "bettertube-local-dev-secret"

init_db()


@app.template_filter("duration")
def duration_filter(seconds: int) -> str:
    total = max(0, int(seconds or 0))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{sec:02d}"
    return f"{minutes}:{sec:02d}"


def _render_home(
    suggestions: list[dict[str, Any]] | None = None,
    suggested_count: int = 3,
):
    if suggestions is None:
        active_suggestions, active_count = get_active_suggestions()
        if active_suggestions:
            suggestions = active_suggestions
            suggested_count = active_count

    trained_videos = list_trained_videos(limit=20)
    return render_template(
        "index.html",
        trained_videos=trained_videos,
        trained_count=get_video_count(),
        suggestions=suggestions,
        suggested_count=suggested_count,
        feedback_map=get_recommendation_feedback_map(),
    )


@app.route("/", methods=["GET"])
def index():
    return _render_home()


@app.post("/train")
def train():
    youtube_url = (request.form.get("youtube_url") or "").strip()
    if not youtube_url:
        flash("Please provide a YouTube video link.", "error")
        return redirect(url_for("index"))

    try:
        added, video = train_video(youtube_url)
        if added:
            flash(f"Trained on '{video['title']}' by {video['creator']}.", "success")
        else:
            flash(f"Video already trained: '{video['title']}'.", "info")
    except Exception as exc:
        flash(f"Training failed: {exc}", "error")

    return redirect(url_for("index"))


@app.post("/train-file")
def train_file():
    uploaded = request.files.get("links_file")
    if uploaded is None or not uploaded.filename:
        flash("Please choose a text file with YouTube links.", "error")
        return redirect(url_for("index"))

    raw = uploaded.read()
    if not raw:
        flash("Uploaded file is empty.", "error")
        return redirect(url_for("index"))

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1", errors="ignore")

    summary = train_videos_from_text(text)
    if summary["total_links"] == 0:
        flash("No YouTube links found in that file.", "error")
        return redirect(url_for("index"))

    flash(
        (
            f"File processed: {summary['total_links']} links. "
            f"Added: {summary['added_count']}, "
            f"already known: {summary['already_known_count']}, "
            f"failed: {summary['failed_count']}."
        ),
        "success",
    )

    for failure in summary["failed"][:3]:
        flash(f"Skipped {failure['url']}: {failure['error']}", "info")

    return redirect(url_for("index"))


@app.post("/feedback")
def feedback():
    action = (request.form.get("feedback") or "").strip().lower()
    feedback_value = 1 if action == "like" else -1 if action == "dislike" else 0
    if feedback_value == 0:
        flash("Invalid feedback action.", "error")
        return redirect(url_for("index"))

    video_id = (request.form.get("video_id") or "").strip()
    url = (request.form.get("url") or "").strip()
    title = (request.form.get("title") or "").strip()
    creator = (request.form.get("creator") or "").strip()
    duration_seconds_raw = (request.form.get("duration_seconds") or "0").strip()
    try:
        duration_seconds = int(duration_seconds_raw)
    except ValueError:
        duration_seconds = 0

    if not video_id or not url:
        flash("Missing recommendation metadata for feedback.", "error")
        return redirect(url_for("index"))

    metadata: dict[str, Any] | None = None
    try:
        metadata = extract_video_metadata(url)
    except Exception:
        metadata = None

    feedback_title = (metadata or {}).get("title") or title or video_id
    feedback_creator = (metadata or {}).get("creator") or creator or "Unknown creator"
    feedback_duration = int((metadata or {}).get("duration_seconds") or duration_seconds or 0)
    feedback_description = str((metadata or {}).get("description") or "")
    feedback_context = str((metadata or {}).get("context_text") or "")
    feedback_tags = list((metadata or {}).get("tags") or [])
    feedback_subjects = list((metadata or {}).get("subjects") or [])

    try:
        set_recommendation_feedback(
            video_id=video_id,
            url=url,
            title=feedback_title,
            creator=feedback_creator,
            feedback=feedback_value,
            duration_seconds=feedback_duration,
            description=feedback_description,
            context_text=feedback_context,
            tags=feedback_tags,
            subjects=feedback_subjects,
        )
    except Exception as exc:
        flash(f"Could not save feedback: {exc}", "error")
        return redirect(url_for("index"))

    if feedback_value == 1:
        try:
            added, _ = train_video_from_suggestion(
                video_id=video_id,
                url=url,
                title=feedback_title,
                creator=feedback_creator,
                duration_seconds=duration_seconds,
                metadata=metadata,
            )
            if added:
                flash(f"Liked and added '{feedback_title}' to training log.", "success")
            else:
                flash(f"Liked '{feedback_title}' (already in training log).", "info")
        except Exception as exc:
            flash(f"Saved like, but could not add to training log: {exc}", "error")
    else:
        flash(f"Saved feedback for '{feedback_title}' ({action}).", "success")

    active_suggestions, target_count = get_active_suggestions()
    target_count = max(1, min(10, int(target_count or 3)))
    remaining = [item for item in active_suggestions if str(item.get("video_id") or "") != video_id]

    set_active_suggestions(remaining[:target_count], target_count)
    return redirect(url_for("index"))


@app.post("/clear-recommendations")
def clear_recommendations_route():
    clear_feedback = (request.form.get("clear_feedback") or "").strip() == "1"
    clear_recommendations(clear_feedback=clear_feedback)
    if clear_feedback:
        flash("Recommendation logs and feedback were cleared.", "success")
    else:
        flash("Recommendation logs were cleared.", "success")
    return redirect(url_for("index"))


@app.post("/suggest")
def suggest():
    requested = (request.form.get("suggest_count") or "3").strip()
    try:
        suggest_count = int(requested)
    except ValueError:
        suggest_count = 3
    suggest_count = max(1, min(10, suggest_count))

    try:
        suggestions = suggest_videos(count=suggest_count)
    except Exception as exc:
        flash(f"Suggestion failed: {exc}", "error")
        suggestions = []

    if not suggestions:
        flash("No suggestions found yet. Add more training videos and try again.", "info")

    set_active_suggestions(suggestions, suggest_count)
    return _render_home(suggestions=suggestions, suggested_count=suggest_count)


if __name__ == "__main__":
    app.run(debug=True)

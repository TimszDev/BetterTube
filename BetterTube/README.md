# BetterTube

Local Python web app that learns your YouTube watch preferences from links you provide, then suggests 1-10 similar videos.

## Features
- `train` flow:
  - Enter a YouTube link.
  - Or upload a `.txt` file with many YouTube links to bulk train in one shot.
  - App opens the video metadata and stores:
    - title
    - creator/channel
    - duration
    - description
    - tags
    - subject signals (categories or inferred keywords)
- `suggest` flow:
  - Builds a preference profile from all trained videos.
  - Searches YouTube for similar candidates.
  - Scores candidates by metadata similarity and returns top 1-10.
  - You can `like` or `dislike` each suggestion to fine-tune future results.
  - Liked suggestions are added to your training log and removed from the active suggestion list.
  - Disliked suggestions are removed and used as negative training signals (title, creator, tags, description/captions context).
  - Remaining suggestions stay visible and are not auto-filled after you make a decision.
  - You can clear recommendation logs (and optionally clear feedback).
- Metadata depth:
  - Pulls title, creator, duration, tags, description, subjects.
  - Also tries chapter titles and closed-caption/auto-caption text to build `context_text` when descriptions are thin.
- Local SQLite database (`bettertube.db`) for trained videos and recommendation logs.

## Setup
```powershell
cd BetterTube
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open: http://127.0.0.1:5000

## Notes
- The app uses `yt-dlp` to fetch metadata, so internet access is required.
- Some videos may block metadata extraction depending on region, age restrictions, or YouTube anti-bot checks.

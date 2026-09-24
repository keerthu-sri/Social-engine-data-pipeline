#!/usr/bin/env python3
"""
Data Vortex A'26 - Round 3 : live data collector
Topic 7 - Public Reaction to a Celebrity / Influencer Controversy

Sources (each is optional; skipped if credentials are missing):
  reddit    posts + comments      (PRAW, free OAuth app)
  youtube   video comments        (YouTube Data API v3, free key)
  bluesky   public posts          (app password)
  gdelt     news headlines        (no key) -> used to explain *why* sentiment/activity shifted

Every record is normalised to one schema so the notebook can treat all sources alike.
Authors are salted+hashed (no raw usernames stored).

Examples
  # one-off pull
  python scraper.py --queries "Person Name" "Person Name controversy" --sources reddit youtube gdelt

  # true live mode: poll every 10 min for 48 h, appending only new records
  python scraper.py --queries "Person Name" --sources reddit youtube bluesky gdelt \
                    --loop-minutes 10 --duration-hours 48
"""
import argparse, hashlib, json, logging, os, re, time
from datetime import datetime, timezone

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SCHEMA = ["source", "kind", "post_id", "parent_id", "created_utc", "collected_utc",
          "author_hash", "text", "likes", "replies", "shares", "url", "query"]

log = logging.getLogger("scraper")
SALT = os.getenv("AUTHOR_SALT", "change-me")
UA = {"User-Agent": "datavortex-r3-scraper/1.0"}


# ----------------------------------------------------------------- helpers
def utc_iso(ts):
    """epoch seconds | ISO string | datetime -> 'YYYY-MM-DDTHH:MM:SSZ' (UTC)."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    elif isinstance(ts, datetime):
        dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    else:
        s = str(ts)
        if re.fullmatch(r"\d{8}T\d{6}Z", s):                      # GDELT format
            dt = datetime.strptime(s, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        else:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hash_author(name):
    return hashlib.sha256((SALT + str(name)).encode()).hexdigest()[:12] if name else None


def now_iso():
    return utc_iso(datetime.now(timezone.utc))


def rec(source, kind, post_id, created, text, query, author=None, parent_id=None,
        likes=0, replies=0, shares=0, url=None):
    return dict(source=source, kind=kind, post_id=str(post_id), parent_id=parent_id,
                created_utc=utc_iso(created), collected_utc=now_iso(),
                author_hash=hash_author(author), text=(text or "").strip(),
                likes=likes or 0, replies=replies or 0, shares=shares or 0,
                url=url, query=query)


def get_json(url, params=None, headers=None, retries=4):
    """GET with exponential back-off on 429 / 5xx. Non-retryable errors (400/401/403 - bad
    key, quota exceeded, malformed query) fail fast and log the response body so the real
    cause is visible instead of silently returning None after wasted retries."""
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers={**UA, **(headers or {})}, timeout=30)
            if r.status_code in (429, 500, 502, 503, 504):
                wait = 2 ** i * 3
                log.warning("HTTP %s from %s - retrying in %ss", r.status_code, url, wait)
                time.sleep(wait)
                continue
            if r.status_code in (400, 401, 403):
                log.error("HTTP %s from %s - not retrying. Response: %s", r.status_code, url, r.text[:300])
                return None
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            log.warning("request failed (%s) attempt %d/%d", e, i + 1, retries)
            time.sleep(2 ** i)
    return None


# ----------------------------------------------------------------- sources
def collect_reddit(queries, limit=100, max_comments=200, time_filter="week", subreddits=("all",), **_):
    cid, sec = os.getenv("REDDIT_CLIENT_ID"), os.getenv("REDDIT_CLIENT_SECRET")
    if not (cid and sec):
        log.info("reddit: no credentials, skipped"); return []
    import praw
    reddit = praw.Reddit(client_id=cid, client_secret=sec,
                         user_agent=os.getenv("REDDIT_USER_AGENT", UA["User-Agent"]))
    out = []
    for q in queries:
        for sub in subreddits:
            try:
                for s in reddit.subreddit(sub).search(q, sort="new", time_filter=time_filter, limit=limit):
                    body = f"{s.title}. {s.selftext}" if s.selftext else s.title
                    out.append(rec("reddit", "post", s.id, s.created_utc, body, q,
                                   author=getattr(s.author, "name", None), likes=s.score,
                                   replies=s.num_comments, url=f"https://reddit.com{s.permalink}"))
                    if max_comments and s.num_comments:
                        s.comment_sort = "new"
                        s.comments.replace_more(limit=0)
                        for c in s.comments.list()[:max_comments]:
                            out.append(rec("reddit", "comment", c.id, c.created_utc, c.body, q,
                                           author=getattr(c.author, "name", None),
                                           parent_id=s.id, likes=c.score,
                                           url=f"https://reddit.com{c.permalink}"))
            except Exception as e:
                log.warning("reddit error on %r/%r: %s", sub, q, e)
    log.info("reddit: %d records", len(out)); return out


def collect_youtube(queries, max_videos=12, max_comments=500, since=None, order="relevance", **_):
    key = os.getenv("YOUTUBE_API_KEY")
    if not key:
        log.info("youtube: no API key, skipped"); return []
    base, out = "https://www.googleapis.com/youtube/v3", []
    for q in queries:
        # "relevance" surfaces established videos with accumulated comments; "date" finds only
        # just-uploaded videos, which usually have few or zero comments yet - use "date" only
        # once you're deliberately hunting for brand-new coverage of a fresh trigger event.
        params = dict(part="snippet", q=q, type="video", order=order, maxResults=max_videos, key=key)
        if since:
            params["publishedAfter"] = since
        vids = get_json(f"{base}/search", params) or {}
        if not vids.get("items"):
            log.warning("youtube: 0 videos found for query %r", q)
        for v in vids.get("items", []):
            vid, title = v["id"]["videoId"], v["snippet"]["title"]
            token, n = None, 0
            while n < max_comments:
                p = dict(part="snippet", videoId=vid, maxResults=100, order="time",
                         textFormat="plainText", key=key)
                if token:
                    p["pageToken"] = token
                data = get_json(f"{base}/commentThreads", p)
                if not data:
                    break           # comments disabled / quota exhausted
                for it in data.get("items", []):
                    c = it["snippet"]["topLevelComment"]["snippet"]
                    out.append(rec("youtube", "comment", it["id"], c["publishedAt"], c["textOriginal"], q,
                                   author=(c.get("authorChannelId") or {}).get("value"),
                                   parent_id=vid, likes=c.get("likeCount"),
                                   replies=it["snippet"].get("totalReplyCount"),
                                   url=f"https://youtube.com/watch?v={vid}&lc={it['id']}"))
                    n += 1
                token = data.get("nextPageToken")
                if not token:
                    break
            if n == 0:
                log.info("youtube: 0 comments for video %r (%s) - likely comments disabled", title, vid)
            else:
                log.info("youtube: %d comments from video %r", n, title)
    log.info("youtube: %d records total", len(out)); return out


def collect_bluesky(queries, limit=100, pages=5, **_):
    h, pw = os.getenv("BSKY_HANDLE"), os.getenv("BSKY_APP_PASSWORD")
    if not (h and pw):
        log.info("bluesky: no credentials, skipped"); return []
    try:
        r = requests.post("https://bsky.social/xrpc/com.atproto.server.createSession",
                          json={"identifier": h, "password": pw}, timeout=30)
        r.raise_for_status()
        hdr = {"Authorization": f"Bearer {r.json()['accessJwt']}"}
    except Exception as e:
        log.warning("bluesky login failed: %s", e); return []
    out = []
    for q in queries:
        cursor = None
        for _p in range(pages):
            p = dict(q=q, sort="latest", limit=limit)
            if cursor:
                p["cursor"] = cursor
            data = get_json("https://bsky.social/xrpc/app.bsky.feed.searchPosts", p, hdr)
            if not data:
                break
            for post in data.get("posts", []):
                rk = post["uri"].split("/")[-1]
                out.append(rec("bluesky", "post", post["uri"], post["record"].get("createdAt"),
                               post["record"].get("text"), q, author=post["author"]["did"],
                               likes=post.get("likeCount"), replies=post.get("replyCount"),
                               shares=(post.get("repostCount") or 0) + (post.get("quoteCount") or 0),
                               url=f"https://bsky.app/profile/{post['author']['handle']}/post/{rk}"))
            cursor = data.get("cursor")
            if not cursor:
                break
    log.info("bluesky: %d records", len(out)); return out


def collect_gdelt(queries, timespan="7d", max_records=250, gdelt_sleep=12.0, gdelt_lang="english", **_):
    """News headlines (no key). gdelt_lang: "english" (default), "tamil", or "any" (no language
    filter - mixes all languages GDELT indexed, useful since a lot of the real coverage here
    is Tamil-language press, not English)."""
    out = []
    lang_clause = "" if gdelt_lang == "any" else f" sourcelang:{gdelt_lang}"
    for q in queries:
        data = get_json("https://api.gdeltproject.org/api/v2/doc/doc",
                        dict(query=f'"{q}"{lang_clause}', mode="artlist", format="json",
                             maxrecords=max_records, timespan=timespan, sort="datedesc"),
                        retries=2)
        for a in (data or {}).get("articles", []):
            out.append(rec("gdelt_news", "news", hashlib.md5(a["url"].encode()).hexdigest(),
                           a["seendate"], a.get("title"), q, author=a.get("domain"), url=a["url"]))
        time.sleep(gdelt_sleep)
    log.info("gdelt: %d records", len(out)); return out


SOURCES = dict(reddit=collect_reddit, youtube=collect_youtube,
               bluesky=collect_bluesky, gdelt=collect_gdelt)


# ----------------------------------------------------------------- storage
def merge_and_save(new_rows, out_prefix, min_chars=10):
    df_new = pd.DataFrame(new_rows, columns=SCHEMA)
    path_csv = f"{out_prefix}.csv"
    if os.path.exists(path_csv):
        df = pd.concat([pd.read_csv(path_csv, dtype={"post_id": str}), df_new], ignore_index=True)
    else:
        df = df_new
    before = len(df)
    df = df[df["text"].fillna("").str.len() >= min_chars]
    df = df.dropna(subset=["created_utc"])
    df = df.drop_duplicates(subset=["source", "post_id"], keep="last")
    df = df.sort_values("created_utc").reset_index(drop=True)
    df.to_csv(path_csv, index=False, encoding="utf-8-sig")  # -sig adds a BOM so Excel
    # auto-detects UTF-8 instead of mojibake-ing Tamil/other non-Latin text on open
    df.to_json(f"{out_prefix}.json", orient="records", indent=2, force_ascii=False)
    log.info("saved %d unique records (%d before cleaning) -> %s(.csv/.json)", len(df), before, out_prefix)
    return df


# ----------------------------------------------------------------- main
def run_once(args):
    rows = []
    for name in args.sources:
        try:
            rows += SOURCES[name](args.queries, subreddits=args.subreddits,
                                  time_filter=args.reddit_window, since=args.since,
                                  timespan=args.gdelt_timespan, gdelt_sleep=args.gdelt_sleep,
                                  gdelt_lang=args.gdelt_lang,
                                  max_videos=args.yt_max_videos, max_comments=args.yt_max_comments,
                                  order=args.yt_order)
        except Exception as e:                       # one bad source must not kill the run
            log.error("%s failed: %s", name, e)
    return merge_and_save(rows, args.out, args.min_chars)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", nargs="+", required=True,
                    help='search terms, e.g. "Name" "Name apology" "#NameCancelled"')
    ap.add_argument("--sources", nargs="+", default=["reddit", "youtube", "gdelt"], choices=SOURCES)
    ap.add_argument("--subreddits", nargs="+", default=["all"])
    ap.add_argument("--reddit-window", default="week", choices=["hour", "day", "week", "month", "year", "all"])
    ap.add_argument("--since", default=None, help="YouTube only: ISO time e.g. 2026-09-15T00:00:00Z")
    ap.add_argument("--gdelt-timespan", default="7d", help="e.g. 24h, 7d, 2w")
    ap.add_argument("--gdelt-sleep", type=float, default=12.0, help="seconds between GDELT queries")
    ap.add_argument("--gdelt-lang", default="english", help="'english', 'tamil', or 'any' (no filter)")
    ap.add_argument("--yt-max-videos", type=int, default=12, help="videos fetched per query")
    ap.add_argument("--yt-max-comments", type=int, default=500, help="max comments fetched per video")
    ap.add_argument("--yt-order", default="relevance", choices=["relevance", "date", "rating", "viewCount"],
                    help="'relevance' finds established videos with comments already accumulated; "
                         "'date' finds only just-uploaded videos, often with few/no comments yet")
    ap.add_argument("--out", default="datasets/live_processing", help="output prefix -> .csv and .json")
    ap.add_argument("--min-chars", type=int, default=10)
    ap.add_argument("--loop-minutes", type=float, default=0, help=">0 enables live polling")
    ap.add_argument("--duration-hours", type=float, default=24)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.loop_minutes <= 0:
        run_once(args); return
    end = time.time() + args.duration_hours * 3600
    while time.time() < end:
        run_once(args)
        time.sleep(args.loop_minutes * 60)


if __name__ == "__main__":
    main()
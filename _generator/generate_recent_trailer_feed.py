#!/usr/bin/env python3
"""Generate Watched's public movie-trailer feed without a YouTube API key."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import pathlib
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from typing import Any, TypeVar


DEFAULT_TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_BASE_URL = os.environ.get("TMDB_BASE_URL", DEFAULT_TMDB_BASE_URL).rstrip("/")
YOUTUBE_OEMBED_URL = "https://www.youtube.com/oembed"
YOUTUBE_FEED_URL = "https://www.youtube.com/feeds/videos.xml"
DEFAULT_CHANNELS_PATH = pathlib.Path(__file__).with_name("recent-trailer-channels.json")
USER_AGENT = "WatchedTrailerFeed/1.0 (+https://github.com/codesmithing-dev/watched-trailer-feed)"
ATOM_NAMESPACE = "http://www.w3.org/2005/Atom"
YOUTUBE_NAMESPACE = "http://www.youtube.com/xml/schemas/2015"
TRAILER_MARKER = re.compile(
    r"\b(?:(?:official|new|first|final|full)\s+)*(?:teaser(?:\s+trailer)?|trailer)\b",
    re.IGNORECASE,
)
NON_MOVIE_MARKER = re.compile(
    r"\b(?:season\s*\d+|episode\s*\d+|trailer\s+reaction|trailer\s+breakdown|"
    r"fan[ -]?made|concept\s+trailer|gameplay\s+trailer|launch\s+trailer)\b",
    re.IGNORECASE,
)
TRAILER_QUALIFIER_PARTS = {
    "english",
    "green band",
    "hindi",
    "international",
    "red band",
    "tamil",
    "telugu",
    "uk",
    "us",
}
CHANNEL_ID_IN_URL = re.compile(r"/channel/(UC[A-Za-z0-9_-]+)")
CHANNEL_ID_IN_PAGE = re.compile(r'"channelId":"(UC[A-Za-z0-9_-]+)"')
T = TypeVar("T")
R = TypeVar("R")


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def tmdb_token_is_required(base_url: str) -> bool:
    return urllib.parse.urlparse(base_url).hostname == "api.themoviedb.org"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the recent-trailer JSON feed consumed by Watched."
    )
    parser.add_argument("--pages", type=int, default=100, help="TMDB pages per movie list")
    parser.add_argument("--max-movies", type=int, default=2000)
    parser.add_argument("--max-trailers", type=int, default=100)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--language", default="en-US")
    parser.add_argument("--region", default="US")
    parser.add_argument("--channel-lookback-days", type=int, default=45)
    parser.add_argument(
        "--channel-discovery-videos",
        type=int,
        default=500,
        help="Newest official TMDB videos checked for previously unknown channels",
    )
    parser.add_argument("--channels", type=pathlib.Path, default=DEFAULT_CHANNELS_PATH)
    parser.add_argument("--previous-feed", type=pathlib.Path)
    parser.add_argument(
        "--skip-channel-polling",
        action="store_true",
        help="Generate from TMDB only (intended for diagnostics)",
    )
    arguments = parser.parse_args()
    if not 1 <= arguments.pages <= 100:
        parser.error("--pages must be from 1 through 100")
    if not 1 <= arguments.max_movies <= 2000:
        parser.error("--max-movies must be from 1 through 2000")
    if not 1 <= arguments.max_trailers <= 1000:
        parser.error("--max-trailers must be from 1 through 1000")
    if not 1 <= arguments.workers <= 32:
        parser.error("--workers must be from 1 through 32")
    if not 1 <= arguments.channel_lookback_days <= 180:
        parser.error("--channel-lookback-days must be from 1 through 180")
    if not 1 <= arguments.channel_discovery_videos <= 2000:
        parser.error("--channel-discovery-videos must be from 1 through 2000")
    return arguments


def request_bytes(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    bearer_token: str | None = None,
    attempts: int = 4,
) -> bytes:
    if params:
        query = urllib.parse.urlencode(params)
        url = f"{url}?{query}"
    headers = {"Accept": "application/json, application/atom+xml;q=0.9, */*;q=0.8", "User-Agent": USER_AGENT}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    request = urllib.request.Request(url, headers=headers)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in {429, 500, 502, 503, 504} or attempt + 1 == attempts:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
        except (TimeoutError, urllib.error.URLError) as error:
            last_error = error
            if attempt + 1 == attempts:
                raise
            delay = 2**attempt
        time.sleep(delay)
    assert last_error is not None
    raise last_error


def request_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    bearer_token: str | None = None,
) -> dict[str, Any]:
    return json.loads(request_bytes(url, params=params, bearer_token=bearer_token))


def parallel_collect(
    values: Iterable[T],
    operation: Callable[[T], R],
    *,
    workers: int,
    label: str,
    max_failure_fraction: float,
) -> list[R]:
    items = list(values)
    if not items:
        return []
    results: list[R] = []
    failures: list[tuple[T, Exception]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_values = {executor.submit(operation, value): value for value in items}
        for completed, future in enumerate(concurrent.futures.as_completed(future_values), start=1):
            value = future_values[future]
            try:
                results.append(future.result())
            except Exception as error:  # noqa: BLE001 - aggregate concurrent HTTP failures
                failures.append((value, error))
            if completed % 250 == 0:
                log(f"{label}: completed {completed} of {len(items)}")
    if failures:
        log(f"{label}: {len(failures)} of {len(items)} requests failed")
        if len(failures) / len(items) > max_failure_fraction:
            sample = "; ".join(str(error) for _, error in failures[:3])
            raise RuntimeError(f"{label} exceeded its failure threshold: {sample}")
    return results


def movie_from_tmdb(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "tmdbID": value["id"],
        "movieTitle": value.get("title") or value.get("original_title") or "",
        "originalTitle": value.get("original_title"),
        "overview": value.get("overview"),
        "releaseDate": value.get("release_date"),
        "posterPath": value.get("poster_path"),
        "backdropPath": value.get("backdrop_path"),
        "popularity": value.get("popularity"),
        "voteAverage": value.get("vote_average"),
        "voteCount": value.get("vote_count"),
        "originalLanguage": value.get("original_language"),
    }


def trailer_from_video(
    movie: dict[str, Any],
    video: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    return {
        **movie,
        "videoID": video["key"],
        "trailerName": video.get("name") or "Official Trailer",
        "publishedAt": video["published_at"],
        **extra,
    }


def load_movies(arguments: argparse.Namespace, token: str) -> list[dict[str, Any]]:
    endpoints = ("upcoming", "now_playing", "popular")

    def load_page(job: tuple[str, int]) -> tuple[str, int, dict[str, Any]]:
        endpoint, page = job
        response = request_json(
            f"{TMDB_BASE_URL}/movie/{endpoint}",
            params={
                "page": page,
                "language": arguments.language,
                "region": arguments.region,
            },
            bearer_token=token,
        )
        return endpoint, page, response

    first_pages = parallel_collect(
        [(endpoint, 1) for endpoint in endpoints],
        load_page,
        workers=min(arguments.workers, len(endpoints)),
        label="TMDB movie lists",
        max_failure_fraction=0,
    )
    page_jobs: list[tuple[str, int]] = []
    page_responses = list(first_pages)
    for endpoint, _, response in first_pages:
        last_page = min(arguments.pages, int(response.get("total_pages") or 1))
        page_jobs.extend((endpoint, page) for page in range(2, last_page + 1))
    page_responses.extend(
        parallel_collect(
            page_jobs,
            load_page,
            workers=arguments.workers,
            label="TMDB movie lists",
            max_failure_fraction=0,
        )
    )
    movies_by_id: dict[int, dict[str, Any]] = {}
    for _, _, response in page_responses:
        for value in response.get("results", []):
            if value.get("id"):
                movies_by_id[value["id"]] = movie_from_tmdb(value)
    movies = sorted(
        movies_by_id.values(),
        key=lambda movie: movie.get("releaseDate") or "",
        reverse=True,
    )[: arguments.max_movies]
    log(f"Sampling official videos for {len(movies)} unique movies")
    return movies


def load_tmdb_trailers(
    movies: list[dict[str, Any]],
    *,
    token: str,
    workers: int,
) -> list[dict[str, Any]]:
    def load_videos(movie: dict[str, Any]) -> list[dict[str, Any]]:
        response = request_json(
            f"{TMDB_BASE_URL}/movie/{movie['tmdbID']}/videos",
            bearer_token=token,
        )
        trailers = []
        for video in response.get("results", []):
            if (
                video.get("site") == "YouTube"
                and video.get("type") in {"Trailer", "Teaser"}
                and video.get("official") is True
                and video.get("key")
                and video.get("published_at")
            ):
                trailers.append(trailer_from_video(movie, video, source="tmdb"))
        return trailers

    batches = parallel_collect(
        movies,
        load_videos,
        workers=workers,
        label="TMDB movie videos",
        max_failure_fraction=0.02,
    )
    return [trailer for batch in batches for trailer in batch]


def load_channel_catalog(path: pathlib.Path) -> list[dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schemaVersion") != 1 or not isinstance(document.get("channels"), list):
        raise ValueError(f"Unsupported trailer-channel catalog: {path}")
    return document["channels"]


def normalized_channel_url(value: str) -> str:
    return value.replace("http://", "https://").rstrip("/")


def discover_channels(
    trailers: list[dict[str, Any]],
    static_channels: list[dict[str, Any]],
    *,
    workers: int,
    discovery_video_limit: int,
) -> list[dict[str, Any]]:
    known_by_url = {
        normalized_channel_url(channel["channelURL"]): channel for channel in static_channels
    }
    newest_trailers = sorted(
        trailers,
        key=lambda trailer: trailer["publishedAt"],
        reverse=True,
    )
    unique_videos: dict[str, dict[str, Any]] = {}
    for trailer in newest_trailers:
        unique_videos.setdefault(trailer["videoID"], trailer)
        if len(unique_videos) == discovery_video_limit:
            break

    def load_oembed(item: tuple[str, dict[str, Any]]) -> dict[str, str] | None:
        video_id, _ = item
        try:
            response = request_json(
                YOUTUBE_OEMBED_URL,
                params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
            )
        except urllib.error.HTTPError as error:
            if error.code in {401, 403, 404}:
                return None
            raise
        channel_url = response.get("author_url")
        if not channel_url:
            return None
        return {
            "channelTitle": response.get("author_name") or "",
            "channelURL": normalized_channel_url(channel_url),
            "videoID": video_id,
        }

    discoveries = parallel_collect(
        unique_videos.items(),
        load_oembed,
        workers=workers,
        label="YouTube channel discovery",
        max_failure_fraction=0.25,
    )
    discovered_by_url: dict[str, dict[str, str]] = {}
    for discovery in discoveries:
        if discovery:
            discovered_by_url.setdefault(discovery["channelURL"], discovery)

    unresolved = [
        discovery
        for channel_url, discovery in discovered_by_url.items()
        if channel_url not in known_by_url
    ]

    def resolve_channel(discovery: dict[str, str]) -> dict[str, Any] | None:
        channel_url = discovery["channelURL"]
        path_match = CHANNEL_ID_IN_URL.search(channel_url)
        if path_match:
            channel_id = path_match.group(1)
        else:
            page = request_bytes(channel_url).decode("utf-8", errors="ignore")
            canonical_match = CHANNEL_ID_IN_URL.search(page)
            channel_id = canonical_match.group(1) if canonical_match else ""
        if not channel_id:
            watch_page = request_bytes(
                f"https://www.youtube.com/watch?v={discovery['videoID']}"
            ).decode("utf-8", errors="ignore")
            page_match = CHANNEL_ID_IN_PAGE.search(watch_page)
            channel_id = page_match.group(1) if page_match else ""
        if not channel_id:
            return None
        return {
            "channelTitle": discovery["channelTitle"],
            "channelURL": channel_url,
            "channelID": channel_id,
            "discoveredTrailerCount": 1,
        }

    resolved = parallel_collect(
        unresolved,
        resolve_channel,
        workers=workers,
        label="New YouTube channel identities",
        max_failure_fraction=1,
    )
    channels_by_id = {channel["channelID"]: channel for channel in static_channels}
    for channel in resolved:
        if channel:
            channels_by_id.setdefault(channel["channelID"], channel)
    log(
        f"Polling {len(channels_by_id)} trusted channels "
        f"({len(channels_by_id) - len(static_channels)} discovered this run)"
    )
    return list(channels_by_id.values())


def parse_timestamp(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


def title_is_movie_trailer(title: str, *, current_year: int) -> bool:
    if not TRAILER_MARKER.search(title) or NON_MOVIE_MARKER.search(title):
        return False
    explicit_years = [int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", title)]
    if explicit_years and max(explicit_years) < current_year - 2:
        return False
    return True


def extract_movie_title(title: str) -> str | None:
    marker = TRAILER_MARKER.search(title)
    if not marker:
        return None
    prefix = title[: marker.start()].strip(" \t|:;-–—")
    if prefix:
        parts = re.split(r"\s*\|\s*|\s+[–—-]\s+", prefix)
        candidate = parts[-1].strip()
        if normalized_title(candidate) in TRAILER_QUALIFIER_PARTS and len(parts) > 1:
            candidate = parts[-2].strip()
    else:
        suffix = title[marker.end() :].strip(" \t|:;-–—")
        candidate = re.split(r"\s*\|\s*|\s+[–—-]\s+", suffix, maxsplit=1)[0].strip()
    candidate = re.sub(r"^(?:new|watch)\s+", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s*\((?:19|20)\d{2}(?:\s+movie)?\)\s*$", "", candidate)
    candidate = candidate.strip(" \t'\"“”‘’#")
    return candidate or None


def normalized_title(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value).casefold()
    alphanumeric = re.sub(r"[^a-z0-9]+", " ", decomposed)
    return " ".join(alphanumeric.split())


def title_without_article(value: str) -> str:
    return re.sub(r"^(?:a|an|the)\s+", "", normalized_title(value))


def choose_movie_result(
    candidate: str,
    results: list[dict[str, Any]],
    *,
    now: dt.datetime,
) -> dict[str, Any] | None:
    normalized_candidate = normalized_title(candidate)
    articleless_candidate = title_without_article(candidate)
    exact = []
    for result in results:
        titles = [result.get("title") or "", result.get("original_title") or ""]
        if any(
            normalized_title(title) == normalized_candidate
            or title_without_article(title) == articleless_candidate
            for title in titles
            if title
        ):
            exact.append(result)
    if not exact:
        return None

    def ranking(result: dict[str, Any]) -> tuple[int, float, str]:
        release_date = result.get("release_date") or ""
        is_current = 0
        if release_date:
            try:
                release_year = int(release_date[:4])
                is_current = int(now.year - 1 <= release_year <= now.year + 3)
            except ValueError:
                pass
        return is_current, float(result.get("popularity") or 0), release_date

    return max(exact, key=ranking)


def movie_result_is_current(result: dict[str, Any], *, now: dt.datetime) -> bool:
    release_date = result.get("release_date") or ""
    if len(release_date) < 4:
        return False
    try:
        release_year = int(release_date[:4])
    except ValueError:
        return False
    return now.year - 1 <= release_year <= now.year + 3


def trailer_label(youtube_title: str) -> str:
    lowered = youtube_title.casefold()
    if "final trailer" in lowered:
        return "Final Trailer"
    if "teaser" in lowered:
        return "Official Teaser"
    return "Official Trailer"


def semantic_trailer_label(trailer: dict[str, Any]) -> str:
    value = f"{trailer.get('trailerName') or ''} {trailer.get('youtubeTitle') or ''}".casefold()
    if "final trailer" in value:
        return "final-trailer"
    if "teaser" in value:
        return "teaser"
    numbered = re.search(r"\btrailer\s*(?:#|no\.?\s*)?(\d+)\b", value)
    if numbered:
        return f"trailer-{numbered.group(1)}"
    if "new trailer" in value:
        return "new-trailer"
    return "trailer"


def poll_channel_uploads(
    channels: list[dict[str, Any]],
    *,
    workers: int,
    cutoff: dt.datetime,
    current_year: int,
) -> list[dict[str, Any]]:
    def load_feed(channel: dict[str, Any]) -> list[dict[str, Any]]:
        document = request_bytes(
            YOUTUBE_FEED_URL,
            params={"channel_id": channel["channelID"]},
        )
        root = ET.fromstring(document)
        uploads = []
        for entry in root.findall(f"{{{ATOM_NAMESPACE}}}entry"):
            title = entry.findtext(f"{{{ATOM_NAMESPACE}}}title") or ""
            video_id = entry.findtext(f"{{{YOUTUBE_NAMESPACE}}}videoId") or ""
            published = entry.findtext(f"{{{ATOM_NAMESPACE}}}published") or ""
            if not title or not video_id or not published:
                continue
            if parse_timestamp(published) < cutoff:
                continue
            if not title_is_movie_trailer(title, current_year=current_year):
                continue
            uploads.append(
                {
                    "videoID": video_id,
                    "youtubeTitle": title,
                    "publishedAt": published,
                    "channelTitle": channel.get("channelTitle"),
                    "channelURL": channel.get("channelURL"),
                }
            )
        return uploads

    batches = parallel_collect(
        channels,
        load_feed,
        workers=workers,
        label="YouTube channel feeds",
        max_failure_fraction=0.35,
    )
    return [upload for batch in batches for upload in batch]


def match_channel_uploads(
    uploads: list[dict[str, Any]],
    tmdb_trailers: list[dict[str, Any]],
    *,
    token: str,
    arguments: argparse.Namespace,
    now: dt.datetime,
) -> list[dict[str, Any]]:
    tmdb_by_video = {trailer["videoID"]: trailer for trailer in tmdb_trailers}
    matched: list[dict[str, Any]] = []
    unmatched_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for upload in uploads:
        if upload["videoID"] in tmdb_by_video:
            matched.append({**tmdb_by_video[upload["videoID"]], **upload, "source": "channel+tmdb"})
            continue
        candidate = extract_movie_title(upload["youtubeTitle"])
        if candidate:
            unmatched_by_candidate.setdefault(candidate, []).append(upload)

    def search_movie(candidate: str) -> tuple[str, dict[str, Any]]:
        response = request_json(
            f"{TMDB_BASE_URL}/search/movie",
            params={"query": candidate, "language": arguments.language, "region": arguments.region},
            bearer_token=token,
        )
        return candidate, response

    search_results = parallel_collect(
        unmatched_by_candidate,
        search_movie,
        workers=arguments.workers,
        label="TMDB channel-title matching",
        max_failure_fraction=0.1,
    )
    for candidate, response in search_results:
        result = choose_movie_result(candidate, response.get("results", []), now=now)
        if not result or not movie_result_is_current(result, now=now):
            continue
        movie = movie_from_tmdb(result)
        for upload in unmatched_by_candidate[candidate]:
            matched.append(
                {
                    **movie,
                    **upload,
                    "trailerName": trailer_label(upload["youtubeTitle"]),
                    "source": "channel",
                    "official": True,
                }
            )
    return matched


def load_previous_feed(path: pathlib.Path | None) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schemaVersion") != 1 or not isinstance(document.get("trailers"), list):
        raise ValueError(f"Unsupported previous feed: {path}")
    return document["trailers"]


def merge_trailers(
    previous: list[dict[str, Any]],
    channel: list[dict[str, Any]],
    tmdb: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    by_video_id: dict[str, dict[str, Any]] = {}
    for trailer in [*previous, *channel, *tmdb]:
        video_id = trailer.get("videoID")
        if not video_id:
            continue
        by_video_id[video_id] = {**by_video_id.get(video_id, {}), **trailer}
    release_groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for trailer in by_video_id.values():
        signature = (int(trailer["tmdbID"]), semantic_trailer_label(trailer))
        release_groups.setdefault(signature, []).append(trailer)

    clustered: list[dict[str, Any]] = []
    duplicate_window = dt.timedelta(days=4)
    for group in release_groups.values():
        ascending = sorted(group, key=lambda trailer: parse_timestamp(trailer["publishedAt"]))
        clusters: list[list[dict[str, Any]]] = []
        for trailer in ascending:
            if not clusters:
                clusters.append([trailer])
                continue
            previous_time = parse_timestamp(clusters[-1][-1]["publishedAt"])
            current_time = parse_timestamp(trailer["publishedAt"])
            if current_time - previous_time <= duplicate_window:
                clusters[-1].append(trailer)
            else:
                clusters.append([trailer])
        # Regional copies tend to follow the originating upload. Keeping the
        # earliest item gives the release its real publication date.
        clustered.extend(cluster[0] for cluster in clusters)

    ordered = sorted(
        clustered,
        key=lambda trailer: (parse_timestamp(trailer["publishedAt"]), trailer["videoID"]),
        reverse=True,
    )
    return ordered[:limit]


def main() -> int:
    arguments = parse_args()
    token = os.environ.get("TMDB_READ_ACCESS_TOKEN", "").strip()
    if tmdb_token_is_required(TMDB_BASE_URL) and not token:
        log("TMDB_READ_ACCESS_TOKEN is required when querying TMDB directly.")
        return 2
    now = dt.datetime.now(dt.UTC)
    movies = load_movies(arguments, token)
    tmdb_trailers = load_tmdb_trailers(movies, token=token, workers=arguments.workers)
    previous = load_previous_feed(arguments.previous_feed)
    channel_trailers: list[dict[str, Any]] = []
    channel_count = 0
    upload_count = 0
    if not arguments.skip_channel_polling:
        static_channels = load_channel_catalog(arguments.channels)
        channels = discover_channels(
            tmdb_trailers,
            static_channels,
            workers=arguments.workers,
            discovery_video_limit=arguments.channel_discovery_videos,
        )
        channel_count = len(channels)
        uploads = poll_channel_uploads(
            channels,
            workers=arguments.workers,
            cutoff=now - dt.timedelta(days=arguments.channel_lookback_days),
            current_year=now.year,
        )
        upload_count = len(uploads)
        channel_trailers = match_channel_uploads(
            uploads,
            tmdb_trailers,
            token=token,
            arguments=arguments,
            now=now,
        )
    trailers = merge_trailers(
        previous,
        channel_trailers,
        tmdb_trailers,
        limit=arguments.max_trailers,
    )
    output = {
        "schemaVersion": 1,
        "generatedAt": now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "sourceStats": {
            "tmdbMoviesSampled": len(movies),
            "tmdbTrailers": len(tmdb_trailers),
            "trustedChannelsPolled": channel_count,
            "channelUploadsConsidered": upload_count,
            "channelTrailersMatched": len(channel_trailers),
            "previousTrailersRetainedForMerge": len(previous),
        },
        "trailers": trailers,
    }
    json.dump(output, sys.stdout, ensure_ascii=False, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Minimal GitHub REST API client for MRMiner.

- Token precedence is explicit argument, GITHUB_TOKEN, then project .env.local.
  Unauthenticated use works but is rate-limited (60 core req/hr, 10 search req/min).
- Handles pagination, rate-limit sleeping, and simple retries.
"""

from __future__ import annotations

import os
import signal
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import stat
import threading
import time
import urllib.parse

import requests

API_ROOT = "https://api.github.com"
GRAPHQL_URL = "https://api.github.com/graphql"
DEFAULT_TOKEN_FILE = Path(__file__).resolve().parents[2] / ".env.local"


class GitHubNotFound(Exception):
    pass


class GitHubRequestError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        kind: str,
        status_code: int | None = None,
    ):
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code


class GitHubGraphQLError(GitHubRequestError):
    pass


class InsecureCredentialFile(PermissionError):
    pass


class GitHubWallClockTimeout(TimeoutError):
    pass


@dataclass(frozen=True)
class PaginatedResult:
    items: tuple[dict, ...]
    pages_fetched: int
    complete: bool
    next_page: int | None
    expected_count: int | None


def read_local_github_token(path: str | Path = DEFAULT_TOKEN_FILE) -> str | None:
    """Read GITHUB_TOKEN from a private, regular local file."""
    token_path = Path(path)
    if not token_path.exists():
        return None
    if token_path.is_symlink():
        raise InsecureCredentialFile(f"credential file must not be a symlink: {token_path}")

    file_stat = token_path.stat()
    if not stat.S_ISREG(file_stat.st_mode):
        raise InsecureCredentialFile(
            f"credential path must be a regular file: {token_path}"
        )
    if file_stat.st_uid != os.getuid():
        raise InsecureCredentialFile(
            f"credential file must be owned by the current user: {token_path}"
        )
    if stat.S_IMODE(file_stat.st_mode) & 0o077:
        raise InsecureCredentialFile(
            f"credential file permissions must be 0600 or stricter: {token_path}"
        )

    for raw_line in token_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "GITHUB_TOKEN":
            token = value.strip()
            return token or None
    return None


@contextmanager
def _wall_clock_timeout(seconds: float):
    if (
        not hasattr(signal, "setitimer")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    def raise_timeout(_signum, _frame):
        raise GitHubWallClockTimeout(
            f"GitHub request exceeded {seconds:g}s wall-clock timeout"
        )

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        previous_delay, previous_interval = previous_timer
        if previous_delay > 0:
            elapsed = time.monotonic() - started
            signal.setitimer(
                signal.ITIMER_REAL,
                max(previous_delay - elapsed, 1e-6),
                previous_interval,
            )


class GitHubClient:
    def __init__(
        self,
        token: str | None = None,
        user_agent: str = "MRMiner-S1",
        token_file: str | Path = DEFAULT_TOKEN_FILE,
        request_timeout: float = 30.0,
        request_retries: int = 3,
        session: requests.Session | None = None,
    ):
        if request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if request_retries < 0:
            raise ValueError("request_retries cannot be negative")
        self.token = (
            token
            or os.environ.get("GITHUB_TOKEN")
            or read_local_github_token(token_file)
        )
        self.user_agent = user_agent
        self.request_timeout = request_timeout
        self.request_retries = request_retries
        self._session = session or requests.Session()
        # Search API has a separate, much lower rate limit than the core API.
        self._min_search_interval = 2.5 if self.token else 7.0
        self._last_search_ts = 0.0

    # ------------------------------------------------------------------ http

    def _headers(self) -> dict:
        h = {
            "User-Agent": self.user_agent,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _request(
        self,
        url: str,
        retries: int | None = None,
        *,
        payload: dict | None = None,
        response_kind: str = "json",
    ):
        retries = self.request_retries if retries is None else retries
        if retries < 0:
            raise ValueError("retries cannot be negative")
        if response_kind not in {"json", "bytes"}:
            raise ValueError(f"unsupported response kind: {response_kind}")
        last_err: Exception | None = None
        last_status: int | None = None
        last_kind = "network"
        for attempt in range(retries + 1):
            headers = self._headers()
            try:
                with _wall_clock_timeout(self.request_timeout):
                    response = self._session.request(
                        "POST" if payload is not None else "GET",
                        url,
                        headers=headers,
                        json=payload,
                        timeout=(
                            self.request_timeout,
                            self.request_timeout,
                        ),
                    )
                if response.status_code == 404:
                    response.close()
                    raise GitHubNotFound(url)
                if response.status_code < 400:
                    if response_kind == "bytes":
                        return response.content
                    return response.json()

                last_status = response.status_code
                last_err = requests.HTTPError(
                    f"HTTP {response.status_code}: {url}",
                    response=response,
                )
                remaining = response.headers.get("X-RateLimit-Remaining")
                retry_after = response.headers.get("Retry-After")
                reset = response.headers.get("X-RateLimit-Reset")
                response.close()
                rate_limited = (
                    last_status == 429
                    or remaining == "0"
                    or retry_after is not None
                )
                if rate_limited:
                    last_kind = "rate_limited"
                elif last_status in (401, 403):
                    last_kind = "forbidden"
                else:
                    last_kind = "http"
                if attempt == retries:
                    break
                if rate_limited:
                    if remaining == "0" and reset:
                        wait = min(max(int(reset) - time.time(), 1), 3600)
                        print(f"[github] rate limited, sleeping {wait:.0f}s", flush=True)
                        time.sleep(wait + 1)
                        continue
                    if retry_after is not None:
                        wait = min(max(float(retry_after), 1), 3600)
                        print(
                            f"[github] rate limited, sleeping {wait:.0f}s",
                            flush=True,
                        )
                        time.sleep(wait)
                        continue
                    # secondary rate limit: back off exponentially
                    wait = 30 * (attempt + 1)
                    print(
                        f"[github] HTTP {last_status}, backing off {wait}s",
                        flush=True,
                    )
                    time.sleep(wait)
                    continue
                time.sleep(5 * (attempt + 1))
            except (
                requests.ConnectionError,
                requests.Timeout,
                GitHubWallClockTimeout,
            ) as e:
                last_err = e
                last_kind = "network"
                if attempt == retries:
                    break
                time.sleep(5 * (attempt + 1))
        raise GitHubRequestError(
            f"GitHub request failed after retries: {url}: {last_err}",
            kind=last_kind,
            status_code=last_status,
        )

    # ------------------------------------------------------------------- api

    def get(self, path: str, **params):
        url = f"{API_ROOT}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        return self._request(url)

    def get_pull_diff(self, repository: str, number: int) -> bytes:
        parts = repository.split("/")
        if len(parts) != 2 or not all(parts) or number <= 0:
            raise ValueError("invalid GitHub pull request identity")
        owner, name = (
            urllib.parse.quote(part, safe="")
            for part in parts
        )
        return self._request(
            f"https://github.com/{owner}/{name}/pull/{number}.diff",
            response_kind="bytes",
        )

    def paginate(self, path: str, max_pages: int = 30, **params) -> list:
        """Collect all items of a list endpoint (per_page=100)."""
        return list(
            self.paginate_with_metadata(
                path,
                max_pages=max_pages,
                **params,
            ).items
        )

    def paginate_with_metadata(
        self,
        path: str,
        *,
        max_pages: int = 30,
        expected_count: int | None = None,
        **params,
    ) -> PaginatedResult:
        if max_pages <= 0:
            raise ValueError("max_pages must be positive")
        if expected_count is not None and expected_count < 0:
            raise ValueError("expected_count cannot be negative")
        if expected_count == 0:
            return PaginatedResult((), 0, True, None, 0)
        items: list[dict] = []
        complete = False
        pages_fetched = 0
        for page in range(1, max_pages + 1):
            batch = self.get(path, per_page=100, page=page, **params)
            pages_fetched = page
            if not batch:
                complete = True
                break
            items.extend(batch)
            if (
                len(batch) < 100
                or (
                    expected_count is not None
                    and len(items) >= expected_count
                )
            ):
                complete = True
                break
        return PaginatedResult(
            items=tuple(items),
            pages_fetched=pages_fetched,
            complete=complete,
            next_page=None if complete else pages_fetched + 1,
            expected_count=expected_count,
        )

    def graphql(self, query: str, variables: dict) -> dict:
        if not self.token:
            raise GitHubGraphQLError(
                "GitHub GraphQL requires an authenticated token",
                kind="forbidden",
                status_code=401,
            )
        response = self._request(
            GRAPHQL_URL,
            payload={"query": query, "variables": variables},
        )
        errors = response.get("errors")
        if errors:
            raise GitHubGraphQLError(
                f"GitHub GraphQL returned errors: {errors}",
                kind="graphql",
                status_code=200,
            )
        data = response.get("data")
        if not isinstance(data, dict):
            raise GitHubGraphQLError(
                "GitHub GraphQL response has no data object",
                kind="malformed",
                status_code=200,
            )
        return data

    def search_issues(self, query: str, page: int = 1, per_page: int = 100) -> dict:
        """One page of /search/issues, throttled to respect the search rate limit."""
        gap = time.time() - self._last_search_ts
        if gap < self._min_search_interval:
            time.sleep(self._min_search_interval - gap)
        self._last_search_ts = time.time()
        q = urllib.parse.quote(query, safe="")
        return self._request(
            f"{API_ROOT}/search/issues?q={q}&per_page={per_page}&page={page}"
        )

"""On-disk response cache for ingestion connectors.

Public biomedical APIs are rate-limited, occasionally unavailable, and in some
network environments blocked outright. A durable cache makes development,
testing and reproducibility possible without hammering them: a pipeline re-run
produces byte-identical raw payloads, which is a precondition for reproducing a
historical prediction.

Entries are content-addressed by a hash of (method, URL, params, body) and
stored as a metadata JSON file beside the raw body, so a cached payload can be
inspected with ordinary tools.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any


class ResponseCache:
    def __init__(self, root: Path, namespace: str) -> None:
        self.root = Path(root) / namespace
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(
        method: str,
        url: str,
        params: dict[str, Any] | None,
        json_body: dict[str, Any] | None,
    ) -> str:
        blob = json.dumps(
            {
                "method": method.upper(),
                "url": url,
                "params": params or {},
                "body": json_body or {},
            },
            sort_keys=True, default=str, separators=(",", ":"),
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def _paths(self, key: str) -> tuple[Path, Path]:
        shard = self.root / key[:2]
        shard.mkdir(parents=True, exist_ok=True)
        return shard / f"{key}.meta.json", shard / f"{key}.body"

    def get(self, key: str, *, ttl_hours: int | None) -> dict[str, Any] | None:
        meta_path, body_path = self._paths(key)
        if not meta_path.exists() or not body_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            return None

        if ttl_hours is not None:
            fetched = dt.datetime.fromisoformat(meta["fetched_at"])
            age = dt.datetime.now(dt.UTC) - fetched
            if age > dt.timedelta(hours=ttl_hours):
                return None

        meta["content"] = body_path.read_bytes()
        return meta

    def put(self, key: str, payload) -> None:  # noqa: ANN001
        meta_path, body_path = self._paths(key)
        body_path.write_bytes(payload.content)
        meta_path.write_text(
            json.dumps(
                {
                    "endpoint": payload.endpoint,
                    "params": payload.params,
                    "status_code": payload.status_code,
                    "media_type": payload.media_type,
                    "fetched_at": payload.fetched_at.isoformat(),
                    "content_hash": payload.content_hash,
                },
                indent=2, default=str,
            )
        )

    def clear(self) -> int:
        n = 0
        for p in self.root.rglob("*"):
            if p.is_file():
                p.unlink()
                n += 1
        return n

import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.config import config
from core.time_utils import now_china


class AgentRetentionManager:
    def __init__(self, agent: Any):
        self.agent = agent
        self._last_auto_prune_monotonic = 0.0
        self._last_auto_prune_status: dict[str, Any] = {}
        self._last_auto_prune_check: dict[str, Any] = {}

    def _normalize_days(self, value: Any, default: int) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return default

    def _cutoff_for_days(self, retention_days: int, now: datetime) -> datetime | None:
        if retention_days <= 0:
            return None
        return now - timedelta(days=retention_days)

    def _normalize_limit(self, value: Any) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    def _safe_stat(self, path: Path):
        try:
            return path.stat()
        except OSError:
            return None

    def _modified_sort_key(self, entry: dict[str, Any]) -> tuple[datetime, str]:
        modified_at = entry.get("modified_at") or datetime.min.replace(tzinfo=timezone.utc)
        return modified_at, str(entry.get("path", ""))

    def _collect_expired_entry_paths(
        self,
        entries: list[dict[str, Any]],
        retention_days: int,
        max_items: int,
        max_total_bytes: int,
        now: datetime,
    ) -> tuple[list[Path], dict[str, int]]:
        cutoff = self._cutoff_for_days(retention_days, now)
        expired_paths: list[Path] = []
        expired_markers: set[Path] = set()
        expired_size_map: dict[str, int] = {}

        def mark_expired(entry: dict[str, Any]) -> None:
            path = entry["path"]
            if path in expired_markers:
                return
            expired_markers.add(path)
            expired_paths.append(path)
            expired_size_map[str(path)] = int(entry.get("size", 0) or 0)

        for entry in entries:
            if cutoff is not None and entry.get("modified_at") is not None and entry["modified_at"] < cutoff:
                mark_expired(entry)

        sorted_entries = sorted(entries, key=self._modified_sort_key, reverse=True)
        if max_items > 0 and len(entries) > max_items:
            for entry in sorted_entries[max_items:]:
                mark_expired(entry)

        if max_total_bytes > 0:
            kept_bytes = 0
            kept_any = False
            for entry in sorted_entries:
                if entry["path"] in expired_markers:
                    continue
                entry_size = int(entry.get("size", 0) or 0)
                if not kept_any:
                    kept_bytes += entry_size
                    kept_any = True
                    continue
                if kept_bytes + entry_size <= max_total_bytes:
                    kept_bytes += entry_size
                    continue
                mark_expired(entry)

        return expired_paths, expired_size_map

    def _collect_file_entries(self, root: Path, patterns: tuple[str, ...]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        if not root.exists():
            return entries
        seen: set[Path] = set()
        for pattern in patterns:
            for path in root.glob(pattern):
                if not path.is_file() or path in seen:
                    continue
                seen.add(path)
                stat_result = self._safe_stat(path)
                if stat_result is None:
                    continue
                entries.append(
                    {
                        "path": path,
                        "size": int(stat_result.st_size),
                        "modified_at": datetime.fromtimestamp(stat_result.st_mtime, tz=timezone.utc),
                    }
                )
        return entries

    def _collect_file_target(
        self,
        key: str,
        root: Path,
        retention_days: int,
        max_items: int,
        max_total_bytes: int,
        patterns: tuple[str, ...],
        now: datetime,
    ) -> dict[str, Any]:
        entries = self._collect_file_entries(root, patterns)
        expired_paths, expired_size_map = self._collect_expired_entry_paths(entries, retention_days, max_items, max_total_bytes, now)
        total_bytes = sum(int(entry["size"]) for entry in entries)
        reclaimable_bytes = sum(expired_size_map.values())

        return {
            "key": key,
            "path": str(root),
            "retention_days": retention_days,
            "max_items": max_items,
            "max_total_bytes": max_total_bytes,
            "item_kind": "file",
            "item_count": len(entries),
            "expired_count": len(expired_paths),
            "total_bytes": total_bytes,
            "reclaimable_bytes": reclaimable_bytes,
            "expired_paths": expired_paths,
            "expired_size_map": expired_size_map,
        }

    def _iter_snapshot_request_dirs(self, root: Path) -> list[Path]:
        if not root.exists():
            return []
        return sorted([path for path in root.iterdir() if path.is_dir()], key=lambda item: item.name)

    def _snapshot_dir_stats(self, request_dir: Path) -> tuple[int, int, datetime | None]:
        file_count = 0
        total_bytes = 0
        newest_modified: datetime | None = None
        for path in request_dir.glob("*.json"):
            if not path.is_file():
                continue
            stat_result = self._safe_stat(path)
            if stat_result is None:
                continue
            file_count += 1
            total_bytes += int(stat_result.st_size)
            modified_at = datetime.fromtimestamp(stat_result.st_mtime, tz=timezone.utc)
            if newest_modified is None or modified_at > newest_modified:
                newest_modified = modified_at
        return file_count, total_bytes, newest_modified

    def _collect_snapshot_target(self, root: Path, retention_days: int, max_items: int, max_total_bytes: int, now: datetime) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []

        for request_dir in self._iter_snapshot_request_dirs(root):
            request_file_count, request_bytes, newest_modified = self._snapshot_dir_stats(request_dir)
            entries.append(
                {
                    "path": request_dir,
                    "size": request_bytes,
                    "file_count": request_file_count,
                    "modified_at": newest_modified,
                }
            )

        expired_paths, expired_size_map = self._collect_expired_entry_paths(entries, retention_days, max_items, max_total_bytes, now)

        total_bytes = sum(int(entry["size"]) for entry in entries)
        reclaimable_bytes = sum(expired_size_map.values())
        file_count = sum(int(entry["file_count"]) for entry in entries)

        return {
            "key": "snapshots",
            "path": str(root),
            "retention_days": retention_days,
            "max_items": max_items,
            "max_total_bytes": max_total_bytes,
            "item_kind": "request_dir",
            "item_count": len(entries),
            "file_count": file_count,
            "expired_count": len(expired_paths),
            "total_bytes": total_bytes,
            "reclaimable_bytes": reclaimable_bytes,
            "expired_paths": expired_paths,
            "expired_size_map": expired_size_map,
        }

    def _build_targets(self, now: datetime | None = None) -> list[dict[str, Any]]:
        current_time = now or now_china()
        return [
            self._collect_file_target(
                "logs",
                Path(config.resolve_path(config.log_dir)),
                self._normalize_days(getattr(config, "log_retention_days", 7), 7),
                self._normalize_limit(getattr(config, "log_retention_max_files", 20)),
                self._normalize_limit(getattr(config, "log_retention_max_bytes", 0)),
                ("*.log",),
                current_time,
            ),
            self._collect_snapshot_target(
                Path(config.resolve_path(config.state_snapshot_dir)),
                self._normalize_days(getattr(config, "snapshot_retention_days", 7), 7),
                self._normalize_limit(getattr(config, "snapshot_retention_max_request_dirs", 200)),
                self._normalize_limit(getattr(config, "snapshot_retention_max_bytes", 0)),
                current_time,
            ),
            self._collect_file_target(
                "audit_logs",
                Path(config.resolve_path(getattr(config, "audit_log_dir", os.path.join("runtime_state", "audit")))),
                self._normalize_days(getattr(config, "audit_log_retention_days", 14), 14),
                self._normalize_limit(getattr(config, "audit_log_retention_max_files", 50)),
                self._normalize_limit(getattr(config, "audit_log_retention_max_bytes", 0)),
                ("*.jsonl",),
                current_time,
            ),
            self._collect_file_target(
                "memory_backups",
                Path(config.resolve_path(config.memory_backup_dir)),
                self._normalize_days(getattr(config, "memory_backup_retention_days", 14), 14),
                self._normalize_limit(getattr(config, "memory_backup_retention_max_files", 20)),
                self._normalize_limit(getattr(config, "memory_backup_retention_max_bytes", 0)),
                ("*",),
                current_time,
            ),
        ]

    def get_retention_status(self) -> dict[str, Any]:
        targets = self._build_targets()
        return {
            "generated_at": now_china().isoformat(),
            "targets": [{key: value for key, value in target.items() if key not in {"expired_paths", "expired_size_map"}} for target in targets],
            "last_auto_prune": dict(self._last_auto_prune_status),
            "last_auto_prune_check": dict(self._last_auto_prune_check),
            "totals": {
                "item_count": sum(int(target.get("item_count", 0) or 0) for target in targets),
                "expired_count": sum(int(target.get("expired_count", 0) or 0) for target in targets),
                "total_bytes": sum(int(target.get("total_bytes", 0) or 0) for target in targets),
                "reclaimable_bytes": sum(int(target.get("reclaimable_bytes", 0) or 0) for target in targets),
            },
        }

    def auto_prune_decision(self) -> tuple[bool, dict[str, Any]]:
        checked_at = now_china().isoformat()
        if not bool(getattr(config, "retention_auto_prune_enabled", True)):
            return False, {
                "status": "skipped_disabled",
                "reason": "disabled",
                "checked_at": checked_at,
            }
        min_interval_seconds = self._normalize_limit(getattr(config, "retention_auto_prune_min_interval_seconds", 300))
        if min_interval_seconds <= 0:
            return True, {
                "status": "executed",
                "reason": "ready",
                "checked_at": checked_at,
                "min_interval_seconds": min_interval_seconds,
            }
        now = time.monotonic()
        elapsed_seconds = round(max(0.0, now - self._last_auto_prune_monotonic), 3)
        if elapsed_seconds >= min_interval_seconds:
            return True, {
                "status": "executed",
                "reason": "ready",
                "checked_at": checked_at,
                "min_interval_seconds": min_interval_seconds,
                "elapsed_seconds": elapsed_seconds,
            }
        return False, {
            "status": "skipped_throttled",
            "reason": "throttled",
            "checked_at": checked_at,
            "min_interval_seconds": min_interval_seconds,
            "elapsed_seconds": elapsed_seconds,
        }

    def maybe_auto_prune(self, trigger: str = "") -> dict[str, Any] | None:
        should_run, decision = self.auto_prune_decision()
        decision["trigger"] = trigger or "unspecified"
        self._last_auto_prune_check = dict(decision)
        if not should_run:
            return None
        payload = self.prune_runtime_data(apply=True)
        self._last_auto_prune_monotonic = time.monotonic()
        deleted_count = int((payload.get("totals", {}) or {}).get("deleted_count", 0) or 0)
        self._last_auto_prune_status = {
            "trigger": decision["trigger"],
            "executed_at": now_china().isoformat(),
            "deleted_count": deleted_count,
            "expired_count": int((payload.get("totals", {}) or {}).get("expired_count", 0) or 0),
            "reclaimable_bytes": int((payload.get("totals", {}) or {}).get("reclaimable_bytes", 0) or 0),
        }
        self._last_auto_prune_check = {
            **decision,
            "status": "executed",
            "reason": "cleaned" if deleted_count else "no_expired_items",
            "executed_at": self._last_auto_prune_status["executed_at"],
            "deleted_count": deleted_count,
            "expired_count": self._last_auto_prune_status["expired_count"],
            "reclaimable_bytes": self._last_auto_prune_status["reclaimable_bytes"],
        }
        if deleted_count:
            try:
                from core.llm.manager import llm_manager

                llm_manager.log_structured_event(
                    "retention_auto_prune",
                    message="Automatic retention prune completed",
                    source="retention",
                    outcome="cleaned",
                    trigger=self._last_auto_prune_status["trigger"],
                    deleted_count=deleted_count,
                    reclaimable_bytes=self._last_auto_prune_status["reclaimable_bytes"],
                )
            except Exception:
                pass
        return payload

    def prune_runtime_data(self, apply: bool = False) -> dict[str, Any]:
        targets = self._build_targets()
        cleaned_targets: list[dict[str, Any]] = []
        deleted_count_total = 0
        deleted_bytes_total = 0

        for target in targets:
            deleted_count = 0
            deleted_bytes = 0
            errors: list[str] = []
            expired_size_map = target.get("expired_size_map", {}) or {}
            for path in target.get("expired_paths", []):
                path_size = int(expired_size_map.get(str(path), 0) or 0)
                try:
                    if apply:
                        if path.is_dir():
                            shutil.rmtree(path)
                        elif path.exists():
                            path.unlink()
                    deleted_count += 1
                    deleted_bytes += path_size
                except OSError as exc:
                    errors.append(f"{path}: {exc}")

            cleaned_target = {key: value for key, value in target.items() if key not in {"expired_paths", "expired_size_map"}}
            cleaned_target["deleted_count"] = deleted_count if apply else 0
            cleaned_target["deleted_bytes"] = deleted_bytes if apply else 0
            cleaned_target["mode"] = "apply" if apply else "dry_run"
            if errors:
                cleaned_target["errors"] = errors
            cleaned_targets.append(cleaned_target)
            if apply:
                deleted_count_total += deleted_count
                deleted_bytes_total += deleted_bytes

        return {
            "mode": "apply" if apply else "dry_run",
            "generated_at": now_china().isoformat(),
            "targets": cleaned_targets,
            "totals": {
                "expired_count": sum(int(target.get("expired_count", 0) or 0) for target in cleaned_targets),
                "reclaimable_bytes": sum(int(target.get("reclaimable_bytes", 0) or 0) for target in cleaned_targets),
                "deleted_count": deleted_count_total,
                "deleted_bytes": deleted_bytes_total,
            },
        }
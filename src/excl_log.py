"""src/excl_log.py — Shared exclusion-log utility.

Appends one TSV row per excluded complex to logs/exclusions.tsv so that
every script's hard-exclusion decisions are auditable in one place.
"""
import time
from pathlib import Path


class ExclusionLog:
    HEADER = "timestamp\tscript\tcomplex_id\treason\n"

    def __init__(self, logs_dir: Path, script: str):
        self._path = Path(logs_dir) / "exclusions.tsv"
        self._script = script
        if not self._path.exists():
            self._path.write_text(self.HEADER)

    def drop(self, complex_id: str, reason: str) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(self._path, "a") as fh:
            fh.write(f"{ts}\t{self._script}\t{complex_id}\t{reason}\n")

    def drop_many(self, ids, reason: str) -> int:
        for cid in ids:
            self.drop(cid, reason)
        return len(list(ids)) if not isinstance(ids, (list, tuple, set)) else len(ids)

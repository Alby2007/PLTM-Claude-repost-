"""Tests for the Groq triple-ingestion path (data_ingestion)."""

import sqlite3
import time

from src.analysis.data_ingestion import _filter_triples, _store_triples


class TestFilterTriples:
    def test_placeholder_rows_dropped(self):
        text = "Albert builds PLTM on Windows."
        raw = [
            {"s": "subject", "p": "predicate", "o": "object"},   # prompt echo
            {"s": "entity", "p": "relationship", "o": "x"},
            {"s": "Albert", "p": "builds", "o": "PLTM"},         # grounded
        ]
        kept = _filter_triples(raw, text)
        assert kept == [{"s": "Albert", "p": "builds", "o": "PLTM"}]

    def test_ungrounded_subjects_dropped(self):
        # tinyllama-style echo of the prompt's e.g. examples
        text = "Albert builds PLTM on Windows."
        raw = [{"s": "North_Korea", "p": "tested", "o": "missile"},
               {"s": "Windows", "p": "runs_on", "o": "PC"}]
        assert _filter_triples(raw, text) == [
            {"s": "Windows", "p": "runs_on", "o": "PC"}]

    def test_empty_and_malformed_dropped(self):
        assert _filter_triples([{"s": "", "p": "x", "o": "y"},
                                {"s": "a", "p": "", "o": "y"}], "a x y") == []


class TestStoreTriples:
    def test_safety_screen_and_dedup(self, tmp_path, monkeypatch):
        from src import db_config
        db = tmp_path / "t.db"
        monkeypatch.setattr(db_config, "DB_PATH", db)
        # minimal atoms schema
        conn = sqlite3.connect(str(db))
        conn.execute("""CREATE TABLE atoms (
            id TEXT PRIMARY KEY, atom_type TEXT, graph TEXT,
            subject TEXT, predicate TEXT, object TEXT,
            confidence REAL, first_observed REAL, last_accessed REAL,
            metadata TEXT,
            UNIQUE(subject, predicate, object, graph))""")
        conn.commit(); conn.close()

        triples = [
            {"s": "Albert", "p": "uses", "o": "Windows"},
            {"s": "leak", "p": "has", "o": "password: hunter2"},  # SafetyJudge REJECT
        ]
        r = _store_triples(triples, "test", "x", "text")
        assert r["stored"] == 1
        assert r["skipped_safety"] == 1

        # same triple again -> unique-constraint dupe counted, not error
        r2 = _store_triples(triples[:1], "test", "x", "text")
        assert r2["stored"] == 0
        assert r2["skipped_dupes"] == 1

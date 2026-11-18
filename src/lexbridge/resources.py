"""Builds / loads everything the pipeline needs: statutes, mappings, judgments, chunks, indexes, graph,
transferability. ``load_resources(build=True)`` rebuilds from seed + fixtures (+ data/raw if present)."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from .config import get_config
from .index.chunker import chunk_corpus
from .index.dense import DenseIndex, make_embedder
from .index.graph import build_graph, load_graph, save_graph
from .index.sparse import BM25Index
from .ingest import judgments as J
from .ingest import mappings as M
from .ingest.statutes import StatuteStore, load_jsonl, load_statutes, save_jsonl
from .nlp.code_resolver import CodeResolver, HeadingLexicon
from .schemas import Chunk, Judgment, SectionMapping, StatuteSection
from .transfer.transferability import TransferResult, classify_all

NON_EXPLICIT_RULES = {"window", "dominant", "date", "keyword"}


@dataclass
class Index:
    chunks: list[Chunk]
    bm25_code: BM25Index
    bm25_plain: BM25Index
    dense: DenseIndex

    @classmethod
    def build(cls, chunks: list[Chunk], cfg) -> "Index":
        texts = [c.text for c in chunks]
        ids = [sorted(set(c.section_ids) | set(c.doc_section_ids)) for c in chunks]
        return cls(chunks, BM25Index(texts, ids, code_aware=True), BM25Index(texts), DenseIndex(make_embedder(cfg), texts))


@dataclass
class Resources:
    statutes: list[StatuteSection]
    store: StatuteStore
    mappings: list[SectionMapping]
    table: M.MappingTable
    judgments: list[Judgment]
    lexicon: HeadingLexicon
    resolver: CodeResolver
    transfer: dict[str, TransferResult]
    graph: object
    indexes: dict[str, Index] = field(default_factory=dict)
    conflicts: list[dict] = field(default_factory=list)

    def index(self, variant: str = "role") -> Index:
        """variants: role (default), naive (fixed windows), noresolver (explicit codes only)."""
        if variant not in self.indexes:
            cfg = get_config()
            ch = cfg["chunking"]
            if variant == "noresolver":
                resolver = _NoResolver(self.lexicon, self.resolver.transition)
                chunks = chunk_corpus(self.judgments, resolver, ch["size"], ch["overlap"], True)
            else:
                chunks = chunk_corpus(self.judgments, self.resolver, ch["size"], ch["overlap"], variant != "naive")
            emb = make_embedder(cfg)
            cache = cfg.path("processed") / "embedding_cache.pkl"
            if hasattr(emb, "load_cache") and not emb.cache:
                emb.load_cache(cache)
            self.indexes[variant] = Index.build(chunks, cfg)
            if hasattr(emb, "save_cache"):
                emb.save_cache(cache)
        return self.indexes[variant]


class _NoResolver(CodeResolver):
    def resolve(self, text, doc_date=None, disable=None):
        return super().resolve(text, doc_date, NON_EXPLICIT_RULES)


def ingest(out_dir: Path, use_network: bool = True) -> tuple[list, list, list, list]:
    cfg = get_config()
    raw, seed, fx = cfg.path("raw"), cfg.path("seed"), cfg.path("fixtures")
    real = not cfg.get("sample_mode", True)
    limit = cfg["sample_sizes"]["judgments"]
    if real:
        # Official India Code text first; HF dataset texts only fill sections India Code lacks.
        from .ingest.mappings import parse_hf_records
        hf_rows = []
        for f in sorted((raw / "mappings" / "hf").glob("*.csv")):
            import csv as _csv
            with open(f, newline="", encoding="utf-8") as fh:
                hf_rows += list(_csv.DictReader(fh))
        _, hf_texts = parse_hf_records(hf_rows) if hf_rows else ([], [])
        extra = [StatuteSection(code=t["code"], section=t["section"], title=t["title"],
                                text=" ".join(t["text"].split()), source="hf", verified=False) for t in hf_texts]
        statutes = load_statutes([raw / "statutes"], extra)
        mappings, conflicts = M.build_mappings(seed / "seed_mappings.csv", raw / "mappings", use_network, real=True)
        judgments = (J.load_parquet_judgments(raw / "judgments" / "overthelex", "ECRT", limit)
                     + J.load_parquet_judgments(raw / "ildc" / "iltur", "ILDC", limit)
                     + J.load_parquet_judgments(raw / "judgments" / "injudgements", "INJ", limit)
                     + J.load_ildc(raw / "ildc", limit) + J.load_folder(raw / "judgments", limit))
        judgments = list({j.doc_id: j for j in judgments}.values())[:limit]
        from .ingest.verify_mappings import check, write_report
        rows = check(mappings, StatuteStore(statutes))
        write_report(rows, out_dir / "mapping_verification.csv")
        n = {k: sum(r["status"] == k for r in rows) for k in ("consistent", "suspect", "unchecked")}
        logger.info(f"mapping auto-check: {n}")
    else:
        statutes = load_statutes([fx / "statutes"])  # sample mode: fixtures only, reproducible
        mappings, conflicts = M.merge([M.load_seed(seed / "seed_mappings.csv")])  # seed only: reproducible
        judgments = J.load_jsonl_judgments(fx / "judgments.jsonl")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(statutes, out_dir / "statutes.jsonl")
    save_jsonl(mappings, out_dir / "mappings.jsonl")
    save_jsonl(judgments, out_dir / "judgments.jsonl")
    M.write_conflicts(conflicts, out_dir / "mapping_conflicts.csv")
    logger.info(f"ingested {len(statutes)} sections, {len(mappings)} mappings ({len(conflicts)} conflicts), "
                f"{len(judgments)} judgments")
    return statutes, mappings, judgments, conflicts


def load_resources(build: bool = False, out_dir: Path | None = None, use_network: bool = False) -> Resources:
    cfg = get_config()
    out_dir = out_dir or cfg.path("processed")
    if build or not (out_dir / "judgments.jsonl").exists():
        statutes, mappings, judgments, conflicts = ingest(out_dir, use_network)
    else:
        statutes = load_jsonl(out_dir / "statutes.jsonl", StatuteSection)
        mappings = load_jsonl(out_dir / "mappings.jsonl", SectionMapping)
        judgments = load_jsonl(out_dir / "judgments.jsonl", Judgment)
        conflicts = []
    store = StatuteStore(statutes)
    lexicon = HeadingLexicon.build(statutes, mappings)
    resolver = CodeResolver(lexicon, cfg.transition)
    import os
    rules_only = os.environ.get("LEXBRIDGE_TRANSFER", "") == "rules"
    transfer = classify_all(mappings, store, cache=out_dir / "transferability.jsonl", rules_only=rules_only)
    res = Resources(statutes, store, mappings, M.MappingTable(mappings), judgments, lexicon, resolver,
                    transfer, graph=None, conflicts=conflicts)
    idx_path, g_path = out_dir / "index.pkl", out_dir / "graph.gpickle"
    if not build and idx_path.exists() and g_path.exists():
        with open(idx_path, "rb") as f:
            res.indexes["role"] = pickle.load(f)
        res.graph = load_graph(g_path)
    else:
        idx = res.index("role")
        res.graph = build_graph(statutes, mappings, idx.chunks, {k: v.label for k, v in transfer.items()})
        save_graph(res.graph, g_path)
        with open(idx_path, "wb") as f:
            pickle.dump(idx, f)
        save_jsonl(idx.chunks, out_dir / "chunks.jsonl")
        (out_dir / "index_meta.json").write_text(json.dumps(
            {"chunks": len(idx.chunks), "graph_nodes": res.graph.number_of_nodes(),
             "graph_edges": res.graph.number_of_edges(), "embedder": getattr(idx.dense.embedder, "name", "?")}, indent=2))
    return res

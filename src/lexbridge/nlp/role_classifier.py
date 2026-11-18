"""Learned rhetorical-role tagger (TF-IDF + logistic regression) trained on the MARRO Indian rhetorical-role
corpus (sentence-level). Paragraph role = majority vote of its sentences. Falls back to the rule tagger when no
model file exists.

Training data: data/raw/roles/in/IN-{train,test}-set/*.txt, lines "sentence<TAB>LABEL" (not redistributed:
the source states no license).
python -m lexbridge.nlp.role_classifier   → data/processed*/role_model.pkl + results/*/role_tagger.json
"""
from __future__ import annotations

import json
import pickle
import re
from collections import Counter
from pathlib import Path

from .rhetorical_roles import tag_paragraph

# MARRO / Bhattacharya et al. labels → LexBridge roles
LABEL_MAP = {
    "FAC": "facts", "ARG": "arguments", "ARG_PETITIONER": "arguments", "ARG_RESPONDENT": "arguments",
    "STA": "statute", "PRE": "analysis", "PRE_RELIED": "analysis", "PRE_NOT_RELIED": "analysis",
    "PRECEDENT": "analysis", "RATIO": "analysis", "RLC": "analysis", "ANALYSIS": "analysis",
    "RPC": "decision", "DIS": "decision", "PREAMBLE": "preamble", "ISSUE": "preamble", "NONE": "other",
}
_MODEL = None


def read_split(d: Path) -> list[tuple[str, str]]:
    rows = []
    for f in sorted(d.glob("*.txt")):
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "\t" not in line:
                continue
            sent, lab = line.rsplit("\t", 1)
            lab = LABEL_MAP.get(lab.strip().upper())
            if lab and sent.strip():
                rows.append((sent.strip(), lab))
    return rows


def _sentences(p: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.;?!])\s+", p) if len(s) > 15] or [p]


def load_model(path: Path | None = None):
    global _MODEL
    if _MODEL is None:
        from ..config import get_config
        path = path or get_config().path("processed") / "role_model.pkl"
        _MODEL = pickle.load(open(path, "rb")) if path.exists() else False
    return _MODEL


def tag_paragraph_learned(p: str) -> str:
    m = load_model()
    if not m:
        return tag_paragraph(p)
    votes = Counter(m.predict(_sentences(p)))
    return votes.most_common(1)[0][0]


def macro_f1(gold, pred):
    from sklearn.metrics import f1_score
    return float(f1_score(gold, pred, average="macro"))


def main():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.pipeline import make_pipeline

    from ..config import get_config
    cfg = get_config()
    root = cfg.path("raw") / "roles" / "in"
    train, test = read_split(root / "IN-train-set"), read_split(root / "IN-test-set")
    if not train:
        raise SystemExit(f"no training data under {root}")
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
                          LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0))
    model.fit([s for s, _ in train], [l for _, l in train])
    gold = [l for _, l in test]
    pred_learned = list(model.predict([s for s, _ in test]))
    pred_rules = [tag_paragraph(s) for s, _ in test]
    report = {"train_sentences": len(train), "test_sentences": len(test),
              "label_dist_test": Counter(gold),
              "macroF1_rules": round(macro_f1(gold, pred_rules), 4),
              "macroF1_learned": round(macro_f1(gold, pred_learned), 4),
              "report_learned": classification_report(gold, pred_learned, output_dict=True, zero_division=0)}
    out = cfg.path("processed")
    out.mkdir(parents=True, exist_ok=True)
    pickle.dump(model, open(out / "role_model.pkl", "wb"))
    res = cfg.path("results")
    res.mkdir(parents=True, exist_ok=True)
    (res / "role_tagger.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "report_learned"}, indent=2, default=str))


if __name__ == "__main__":
    main()

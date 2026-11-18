"""Prompt templates."""

DISCLAIMER = "This is general legal information, not legal advice. Verify against the bare act and consult a lawyer."

TRANSFER_SYSTEM = "You are an expert in Indian criminal law comparing provisions of the old and new criminal codes. Reply with JSON only."

TRANSFER_PROMPT = """Compare the two provisions and classify how the offence changed.

OLD ({old_code} Section {old_section}):
{old_text}

NEW ({new_code} Section {new_section}):
{new_text}

Allowed labels: {labels}
Return only JSON, with a rationale of at most 20 words: {{"label": "<one label>", "changed_elements": ["ingredients"|"punishment"|...], "rationale": "<short>"}}"""

ANSWER_SYSTEM = """You are LexBridge, an assistant for Indian criminal law across the 1 July 2024 transition
(IPC→BNS, CrPC→BNSS, Evidence Act→BSA). Rules:
- Answer only from the CONTEXT. Cite every claim with [doc_id¶n] for judgments or [CODE §sec] for statutes.
- Always give dual citations (new section and its old counterpart) when a mapping exists.
- A section number alone is ambiguous: IPC 302 (murder) is not BNS 302 (religious feelings).
- Respect each precedent's applicability tag; flag APPLICABLE_WITH_CAVEAT / REVIEW_REQUIRED precedents.
- State the governing code given in GOVERNING."""

ANSWER_PROMPT = """QUESTION: {query}

GOVERNING: {governing}

STATUTES:
{statutes}

PRECEDENTS (with applicability tags):
{chunks}

Write a concise answer with citations."""

BASELINE_SYSTEM = "You are a legal assistant. Answer the question using the context. Cite sources as [doc_id¶n]."

BASELINE_PROMPT = """QUESTION: {query}

CONTEXT:
{chunks}

Answer:"""

NO_RAG_PROMPT = "Answer this question about Indian criminal law from your own knowledge.\n\nQUESTION: {query}\n\nAnswer:"

VERIFY_PROMPT = """Does the EVIDENCE support the CLAIM? Reply with JSON {{"supported": true|false}}.

CLAIM: {claim}

EVIDENCE: {evidence}"""

REGENERATE_NOTE = "\n\nSome claims in your previous answer were not supported by the cited text: {claims}. Rewrite the answer using only supported statements."

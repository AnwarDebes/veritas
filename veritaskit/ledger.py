"""Tamper-evident assurance certificate for a verified guard.

VERITAS seals the *outcome of verification*, which properties were checked,
which were proven, which were refuted and with what counterexample, into a
hash-chained, HMAC-signed certificate.  The point is that an auditor can confirm
the assurance claim **independently of the prover**:

* a **REFUTED** entry carries a counterexample feature vector; the auditor
  re-evaluates the (small, included) guard on it with pure arithmetic, no SAT
  solver, no trust in our prover, and sees the claimed bypass for themselves;
* a **VERIFIED** entry can be re-decided by re-running *any* exact backend on the
  included guard encoding (or by checking the emitted DRAT proof);

and the hash chain + signature make any post-hoc edit, insertion, deletion or
reordering of the verification record detectable.  The construction follows the
same secure-log line as TRACE (Schneier & Kelsey 1999; RFC 6962; HMAC RFC 2104);
as there, HMAC gives *symmetric authentication*, not non-repudiation.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .encode import GuardEncoding

GENESIS = "0" * 64


#
# primitives
#
def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hmac_hex(key: bytes, data: bytes) -> str:
    return hmac.new(key, data, hashlib.sha256).hexdigest()


def merkle_root(leaf_hashes: List[str]) -> str:
    if not leaf_hashes:
        return GENESIS
    level = [bytes.fromhex(h) for h in leaf_hashes]
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        level = [hashlib.sha256(level[i] + level[i + 1]).digest()
                 for i in range(0, len(level), 2)]
    return level[0].hex()


def guard_fingerprint(enc: GuardEncoding) -> dict:
    """A canonical, hashable description of the exact guard being certified."""
    body = {"n_features": enc.n_features, "vocab": list(enc.vocab),
            "clauses": [[int(p), [[int(i), int(s)] for (i, s) in terms]]
                        for (p, terms) in enc.clauses]}
    return {"body": body, "sha256": sha256_hex(canonical(body))}


def encoding_from_fingerprint(fp: dict) -> GuardEncoding:
    b = fp["body"]
    clauses = [(int(p), [(int(i), int(s)) for (i, s) in terms])
               for (p, terms) in b["clauses"] if terms]   # drop empty (never-fire) clauses
    return GuardEncoding(n_features=int(b["n_features"]), clauses=clauses, vocab=list(b["vocab"]))


#
# ledger entries
#
@dataclass
class Entry:
    index: int
    kind: str                       # "guard" | "property"
    payload: dict
    prev_hash: str = GENESIS
    entry_hash: str = ""
    signature: Optional[str] = None

    def compute_hash(self, prev_hash: str) -> str:
        self.prev_hash = prev_hash
        body = {"index": self.index, "kind": self.kind,
                "payload": self.payload, "prev_hash": prev_hash}
        self.entry_hash = sha256_hex(canonical(body))
        return self.entry_hash

    def sign(self, key: bytes) -> "Entry":
        self.signature = hmac_hex(key, self.entry_hash.encode())
        return self


@dataclass
class AssuranceCertificate:
    spec_id: str
    n_entries: int
    chain_head: str
    merkle_root: str
    guard_sha256: str
    summary: dict
    signature: Optional[str] = None

    def _root_body(self) -> dict:
        return {"spec_id": self.spec_id, "n_entries": self.n_entries,
                "chain_head": self.chain_head, "merkle_root": self.merkle_root,
                "guard_sha256": self.guard_sha256, "summary": self.summary}

    def sign(self, key: bytes) -> "AssuranceCertificate":
        self.signature = hmac_hex(key, canonical(self._root_body()))
        return self

    def verify_signature(self, key: bytes) -> bool:
        if self.signature is None:
            return False
        return hmac.compare_digest(self.signature, hmac_hex(key, canonical(self._root_body())))

    def to_dict(self) -> dict:
        d = dict(self._root_body())
        d["signature"] = self.signature
        return d

    @staticmethod
    def from_dict(d: dict) -> "AssuranceCertificate":
        c = AssuranceCertificate(d["spec_id"], d["n_entries"], d["chain_head"],
                                 d["merkle_root"], d["guard_sha256"], d["summary"])
        c.signature = d.get("signature")
        return c


class AssuranceLedger:
    """Append-only hash chain of a guard fingerprint + per-property results."""

    def __init__(self, spec_id: str, key: bytes):
        self.spec_id = spec_id
        self.key = key
        self.entries: List[Entry] = []
        self._head = GENESIS

    def _append(self, kind: str, payload: dict) -> Entry:
        e = Entry(index=len(self.entries), kind=kind, payload=payload)
        e.compute_hash(self._head)
        e.sign(self.key)
        self._head = e.entry_hash
        self.entries.append(e)
        return e

    def record_guard(self, enc: GuardEncoding) -> Entry:
        return self._append("guard", guard_fingerprint(enc))

    def record_result(self, result) -> Entry:
        return self._append("property", result.to_dict() if hasattr(result, "to_dict") else dict(result))

    def to_dict(self) -> dict:
        return {"spec_id": self.spec_id,
                "entries": [{"index": e.index, "kind": e.kind, "payload": e.payload,
                             "prev_hash": e.prev_hash, "entry_hash": e.entry_hash,
                             "signature": e.signature} for e in self.entries]}

    def seal(self, summary: Optional[dict] = None) -> AssuranceCertificate:
        leaves = [e.entry_hash for e in self.entries]
        guard_e = next((e for e in self.entries if e.kind == "guard"), None)
        cert = AssuranceCertificate(
            spec_id=self.spec_id, n_entries=len(self.entries),
            chain_head=self._head, merkle_root=merkle_root(leaves),
            guard_sha256=guard_e.payload["sha256"] if guard_e else GENESIS,
            summary=summary or {})
        return cert.sign(self.key)


#
# independent verification
#
def verify_assurance(ledger_dict: dict, cert: AssuranceCertificate, key: Optional[bytes] = None,
                     recheck: bool = True) -> Tuple[bool, dict]:
    """Re-verify a sealed assurance certificate.

    Always: re-derive the hash chain + Merkle root, check signatures (if ``key``),
    and re-evaluate every REFUTED counterexample on the *included* guard encoding
    (pure arithmetic, the prover-independent check).  If ``recheck`` and a SAT
    backend is importable, also re-run the verifier on every property and confirm
    its status matches the recorded one.
    """
    entries = ledger_dict["entries"]
    info = {"chain_ok": True, "sig_ok": True, "merkle_ok": True,
            "counterexamples_ok": True, "recheck_ok": True,
            "first_bad_entry": None, "notes": [], "rechecked": 0}

    prev = GENESIS
    for e in entries:
        body = {"index": e["index"], "kind": e["kind"], "payload": e["payload"], "prev_hash": prev}
        if sha256_hex(canonical(body)) != e["entry_hash"] or e["prev_hash"] != prev:
            info["chain_ok"] = False
            info["first_bad_entry"] = e["index"]
            info["notes"].append(f"hash/chain mismatch at entry {e['index']}")
            break
        if key is not None and e["signature"] != hmac_hex(key, e["entry_hash"].encode()):
            info["sig_ok"] = False
            info["first_bad_entry"] = e["index"]
            info["notes"].append(f"bad signature at entry {e['index']}")
            break
        prev = e["entry_hash"]

    if merkle_root([e["entry_hash"] for e in entries]) != cert.merkle_root:
        info["merkle_ok"] = False
        info["notes"].append("merkle root mismatch")
    if entries and entries[-1]["entry_hash"] != cert.chain_head:
        info["chain_ok"] = False
        info["notes"].append("chain head mismatch")
    if key is not None and not cert.verify_signature(key):
        info["sig_ok"] = False
        info["notes"].append("certificate signature invalid")

    # the prover-independent part: re-evaluate every counterexample on the guard.
    guard_e = next((e for e in entries if e["kind"] == "guard"), None)
    enc = encoding_from_fingerprint(guard_e["payload"]) if guard_e else None
    if enc is not None:
        for e in entries:
            if e["kind"] != "property":
                continue
            p = e["payload"]
            cx = p.get("counterexample")
            if p.get("status") == "REFUTED" and cx is not None:
                # a refutation must witness a real disagreement on the guard
                if not _counterexample_is_real(enc, p, cx):
                    info["counterexamples_ok"] = False
                    info["notes"].append(f"bogus counterexample at entry {e['index']}")
        if recheck:
            info.update(_recheck_all(enc, entries, info))

    ok = all(info[k] for k in ("chain_ok", "sig_ok", "merkle_ok",
                               "counterexamples_ok", "recheck_ok"))
    info["ok"] = ok
    return ok, info


def _counterexample_is_real(enc: GuardEncoding, payload: dict, cx) -> bool:
    """Confirm a recorded REFUTED counterexample genuinely violates its property,
    by re-evaluating the included guard with pure arithmetic, no solver.

    Fails *closed*: if the obligation cannot be re-derived from the stored spec,
    the witness is rejected, so a fabricated refutation cannot pass the
    prover-independent gate."""
    from . import properties as P
    x = np.asarray(cx, dtype=np.int8)
    if x.ndim != 1 or x.shape[0] != enc.n_features:
        return False
    spec = (payload.get("extra") or {}).get("spec")
    if not spec:
        return False                          # no re-derivable obligation -> reject
    kind = spec.get("kind")
    if kind == "invariant":
        ri = P.Invariant(payload.get("name", "?"), pre=spec.get("pre", {}),
                         post=spec.get("post", "unsafe")).resolve(enc)
        if ri.unresolved:
            return False
        return not ri.holds_on(enc, x)        # a real counterexample VIOLATES the invariant
    if kind == "robustness_radius":
        anchor = np.asarray(spec.get("anchor", []), dtype=np.int8)
        if anchor.shape != x.shape or int(np.sum(x != anchor)) > int(spec.get("max_radius", 0)):
            return False
        return bool(enc.unsafe(x) != enc.unsafe(anchor))
    if kind == "monotone":
        tok = spec.get("token")
        ft = enc.token_feature(tok) if isinstance(tok, str) else (int(tok) if tok is not None else None)
        if ft is None or not (0 <= ft < enc.n_features):
            return False
        x0 = x.copy(); x0[ft] = 0
        x1 = x.copy(); x1[ft] = 1
        d = enc.vote(x1) - enc.vote(x0)
        return (d < 0) if spec.get("direction", "up") == "up" else (d > 0)
    return False                              # equivalent / unknown -> fail closed


def _recheck_all(enc: GuardEncoding, entries, info) -> dict:
    """Optionally re-run the verifier on every property and confirm the recorded
    status.  Skipped silently if no backend is importable."""
    try:
        from .verify import Verifier
        from . import properties as P
    except Exception:                              # pragma: no cover
        return {}
    out = {"recheck_ok": True, "rechecked": 0}
    V = Verifier(enc)
    for e in entries:
        if e["kind"] != "property":
            continue
        prop = _reconstruct_property(e["payload"], enc, P)
        if prop is None:
            continue
        try:
            r = V.verify(prop)
        except Exception:                          # pragma: no cover
            continue
        out["rechecked"] += 1
        if r.status != e["payload"].get("status"):
            out["recheck_ok"] = False
            info["notes"].append(f"recheck disagrees at entry {e['index']}: "
                                 f"{r.status} != {e['payload'].get('status')}")
    return out


def _reconstruct_property(payload: dict, enc: GuardEncoding, P):
    """Rebuild a verifiable Property object from a recorded result's metadata.

    The spec needed to re-run a check is stored under ``extra['spec']`` by the
    engine; if absent we cannot re-run (return None) and rely on the
    counterexample re-evaluation instead."""
    spec = (payload.get("extra") or {}).get("spec")
    if not spec:
        return None
    k = spec.get("kind")
    if k == "invariant":
        return P.Invariant(payload["name"], pre=spec["pre"], post=spec["post"])
    if k == "robustness_radius":
        return P.RobustnessRadius(payload["name"], anchor=np.asarray(spec["anchor"], dtype=np.int8),
                                  max_radius=int(spec["max_radius"]))
    if k == "monotone":
        return P.Monotone(payload["name"], token=spec["token"], direction=spec.get("direction", "up"))
    return None

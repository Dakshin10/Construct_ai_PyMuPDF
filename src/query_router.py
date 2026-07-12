"""
Query Routing for the Salit Steel RAG DB
MinoriLabs — Salit Steel Agent Practice

Lightweight keyword/intent routing: maps a natural-language query to a Chroma
`where` filter on chunk_type (and optionally doc_type / source_doc_id) so
granular chunks (size_row/pdf_detail) don't drown out structure_overview /
element_summary matches — and vice versa.

doc_type is treated as an OPEN SET: candidate doc filters are discovered from
the metadata actually present in the collection, never from a hardcoded list.
Falls back to unfiltered similarity search when no clear intent matches.
"""

import re

# chunk_type -> keyword patterns. First matching group of types wins;
# multiple matches become an $in filter.
CHUNK_TYPE_KEYWORDS = {
    "structure_overview": [
        r"\bstorey?s?\b", r"\bstories\b", r"\bfloors?\b", r"\bmaterial\b",
        r"total\s+area", r"\barea\b", r"foundation\s+type", r"roof",
        r"\boverview\b", r"type\s+of\s+(?:the\s+)?structure",
        r"drawing\s+types?", r"what\s+kind\s+of\s+(?:building|structure)",
        r"\bheight\b", r"\bwidth\b", r"\blength\s+of\s+(?:the\s+)?building\b",
        # concrete strength / material grade live on the overview chunk, not
        # on element_summary — route strength questions here.
        r"concrete\s+strength", r"\bstrength\b", r"\bMPa\b",
        r"wall\s+type", r"\bcurtain\s+wall\b", r"\bCMU\b",
    ],
    "element_summary": [
        r"\bbeams?\b", r"\bfootings?\b", r"\bpiers?\b", r"\bcolumns?\b",
        r"\bslabs?\b", r"\bstirrups?\b", r"\bcover\b", r"\blap\b",
        r"\bhook\b", r"\bties\b", r"\bstock\s+length\b", r"\belements?\b",
        r"shape\s+type", r"bend(?:ing)?", r"how\s+many\s+(?:beams?|footings?|columns?|piers?|slabs?)",
    ],
    "size_row": [
        r"\b\d{2}M\b", r"\b\d+\s*mm\s+bars?\b", r"linear\s+met(?:re|er)s?",
        r"\bbar\s+size\b", r"\bunit\s+weight\b", r"spacing\s+for\s+\d+M",
    ],
    "pdf_detail": [
        r"\bdrawing\s+[A-Z]?\d+\b", r"\brevision\b", r"\brev\s*\d+\b",
        r"\bR\d{2}\b", r"\bvariance\b", r"\bxls\b", r"on\s+(?:sheet|drawing)\b",
        r"\.pdf\b", r"\bdocument\b", r"\bfile\b",
    ],
    "project_summary": [
        r"total\s+(?:weight|bars)", r"\bproject\b", r"\brevisions?\s+list\b",
        r"whole\s+project", r"across\s+all", r"\bsummary\b",
    ],
}

_COMPILED = {ct: [re.compile(p, re.I) for p in pats]
             for ct, pats in CHUNK_TYPE_KEYWORDS.items()}

# "Enumerate every element" intent (stage 4-6 style). When this fires on an
# element_summary query we must retrieve EVERY element, not just the top-k by
# similarity — otherwise enumeration answers silently drop rows (Fix 3).
_ENUM_INTENT = [re.compile(p, re.I) for p in [
    r"\btabulate\b", r"\bsummary\s+table\b", r"\bregister\b",
    r"\benumerate\b", r"\bfor\s+each\b", r"\beach\s+element\b",
    r"\bper\s+element\b", r"\bone\s+row\b",
    r"\blist\s+(?:out\s+)?(?:every|all|each)\b",
    r"\bevery\s+(?:element|column|footing|pier|beam|slab|mark)",
    r"\ball\s+(?:the\s+)?(?:elements?|columns?|footings?|piers?|beams?|slabs?|marks?)\b",
    # "what/which structural elements are in this structure" is enumerate intent
    r"\b(?:what|which)\s+(?:structural\s+)?elements?\b",
    r"\belements?\s+(?:are\s+)?(?:present|in\s+(?:this|the)\s+structure)\b",
    # per-element availability audit (stage 2) — must cover every element
    r"\bavailability\s+of\b", r"\bpresent\s*/?\s*missing\b",
    r"check\s+(?:the\s+)?availability",
]]

# Never dump an unbounded collection into context.
ENUM_MAX_K = 50


def _is_enumeration(query):
    return any(p.search(query) for p in _ENUM_INTENT)


def _known_doc_metadata(collection, limit=2000):
    """Discover the doc_types and source_doc_ids actually in the DB (open set)."""
    doc_types, source_ids = set(), set()
    try:
        got = collection.get(limit=limit, include=["metadatas"])
        for m in got.get("metadatas") or []:
            for dt in (m.get("doc_type") or "").split(","):
                if dt.strip():
                    doc_types.add(dt.strip())
            for sid in (m.get("source_doc_id") or "").split(","):
                if sid.strip():
                    source_ids.add(sid.strip())
    except Exception:
        pass
    return doc_types, source_ids


def route_query(query, collection=None, doc_type=None, source_doc_id=None):
    """
    Returns (where_filter_or_None, routing_info_dict).

    - chunk_type filter from keyword scoring (highest-scoring type(s) win)
    - doc_type / source_doc_id filter: from explicit args, or discovered by
      matching query words against metadata present in the collection.
    """
    scores = {}
    for ct, pats in _COMPILED.items():
        hits = sum(1 for p in pats if p.search(query))
        if hits:
            scores[ct] = hits

    matched_types = []
    if scores:
        best = max(scores.values())
        matched_types = sorted(t for t, s in scores.items() if s == best)

    # doc-specific filters — open set, discovered from the collection
    matched_doc_type, matched_source = doc_type, source_doc_id
    if collection is not None and (matched_doc_type is None or matched_source is None):
        known_types, known_sources = _known_doc_metadata(collection)
        q_lower = query.lower()
        if matched_doc_type is None:
            for dt in known_types:
                if dt and dt != "aggregate" and dt in q_lower:
                    matched_doc_type = dt
                    break
        if matched_source is None:
            for sid in known_sources:
                stem = sid.rsplit(".", 1)[0].lower()
                if stem and (sid.lower() in q_lower or stem in q_lower):
                    matched_source = sid
                    break

    conditions = []
    if len(matched_types) == 1:
        conditions.append({"chunk_type": matched_types[0]})
    elif len(matched_types) > 1:
        conditions.append({"chunk_type": {"$in": matched_types}})
    if matched_doc_type:
        conditions.append({"doc_type": matched_doc_type})
    if matched_source:
        conditions.append({"source_doc_id": matched_source})

    where = None
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    info = {
        "matched_chunk_types": matched_types,               # top-scoring (primary)
        "all_matched_types": sorted(scores),                # every type with a signal
        "matched_doc_type": matched_doc_type,
        "matched_source_doc_id": matched_source,
        "fallback_unfiltered": where is None,
        "enumerate_intent": _is_enumeration(query),
    }
    return where, info


def _element_chunk_count(collection, doc_type=None, source_doc_id=None, project_id=None):
    """
    Live count of element_summary chunks matching the given scope (never
    hardcoded). Scopes to project_id / doc_type / source when provided so a
    multi-project DB counts only the relevant project's elements.
    """
    conds = [{"chunk_type": "element_summary"}]
    if project_id:
        conds.append({"project_id": project_id})
    if doc_type:
        conds.append({"doc_type": doc_type})
    if source_doc_id:
        conds.append({"source_doc_id": source_doc_id})
    where = conds[0] if len(conds) == 1 else {"$and": conds}
    try:
        got = collection.get(where=where, include=[])
        return len(got.get("ids", []))
    except Exception:
        return 0


def retrieve(query, collection, n_results=6, doc_type=None, source_doc_id=None,
             project_id=None):
    """
    Route + retrieve. Returns (chunks, routing_info) where each chunk is
    {"text", "meta", "distance"}.

    Fix 3: enumeration-intent queries on element_summary raise top-k to the live
    element_summary count in scope (capped at ENUM_MAX_K).

    Gap 2: compound queries that signal MORE THAN ONE chunk_type (e.g. element
    keywords + "storeys"/"material") retrieve each type in parallel with its own
    cap and merge — so a minority type (one structure_overview chunk) isn't
    starved by the majority type in a single ranked top-k.
    """
    where, info = route_query(query, collection, doc_type, source_doc_id)
    total = max(collection.count(), 1)

    # Enumeration cap for element_summary.
    enum_k = n_results
    if info["enumerate_intent"]:
        cnt = _element_chunk_count(
            collection, doc_type=info["matched_doc_type"],
            source_doc_id=info["matched_source_doc_id"], project_id=project_id)
        if cnt:
            enum_k = min(ENUM_MAX_K, max(n_results, cnt))

    # doc/source conditions reused across per-type queries
    extra = []
    if info["matched_doc_type"]:
        extra.append({"doc_type": info["matched_doc_type"]})
    if info["matched_source_doc_id"]:
        extra.append({"source_doc_id": info["matched_source_doc_id"]})

    def _cap(ct):
        if ct == "element_summary":
            return enum_k
        if ct in ("structure_overview", "project_summary"):
            return 2            # context types: a couple of chunks is plenty
        return n_results

    # structure_overview / project_summary are project-level aggregates
    # (doc_type="aggregate"), so a doc_type/source filter inferred from the
    # query (e.g. "structural elements" → doc_type=structural) must NOT be
    # applied to them, or the aggregate chunk gets filtered out entirely.
    _AGGREGATE_TYPES = {"structure_overview", "project_summary"}

    def _where_for(ct):
        conds = [{"chunk_type": ct}]
        if ct not in _AGGREGATE_TYPES:
            conds += extra
        return conds[0] if len(conds) == 1 else {"$and": conds}

    def _emit(res, out, seen):
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            key = m.get("chunk_id") or (m.get("source_doc_id", ""), d[:60])
            if key in seen:
                continue
            seen.add(key)
            out.append({"text": d, "meta": m, "distance": dist})

    intents = info["all_matched_types"]
    merged, seen = [], set()

    if intents:
        for ct in intents:
            res = collection.query(query_texts=[query], n_results=min(_cap(ct), total),
                                    where=_where_for(ct),
                                    include=["documents", "metadatas", "distances"])
            _emit(res, merged, seen)

    # Fallback: no intent matched, or filtered search returned nothing.
    if not merged:
        info["fallback_unfiltered"] = True
        res = collection.query(query_texts=[query], n_results=min(n_results, total),
                                include=["documents", "metadatas", "distances"])
        _emit(res, merged, seen)

    merged.sort(key=lambda c: c["distance"])
    info["n_results_used"] = len(merged)
    info["retrieved_types"] = sorted({c["meta"].get("chunk_type", "?") for c in merged})
    return merged, info

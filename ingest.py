# ingest.py — local embeddings (MiniLM) + Chroma
# - Embeds Title/Short/FullText PLUS human-readable metadata tags (fixes generic queries like "Access to information")
# - Flat metadata only (Chroma requirement)
# - Force-unique IDs (include row index)
import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3

import os
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# --- Config ---
CSV_URL = "https://docs.google.com/spreadsheets/d/15B3FBIjsfBAAe0oaeRi9MEfFzKa9TfjU7lNVRUOc3-M/export?format=csv&gid=1654752912"
PERSIST_DIR = "./chroma-db"
COLLECTION = "ogp_commitments"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast & free

def get_chroma():
    return chromadb.PersistentClient(
        path=PERSIST_DIR,
        settings=Settings(allow_reset=True)
    )

def to_int01(v):
    """Normalize common truthy/falsy/ND values to 0/1."""
    if v is None:
        return 0
    if isinstance(v, (int, float)):
        try:
            return 1 if int(v) != 0 else 0
        except Exception:
            return 0
    s = str(v).strip().lower()
    if s in {"1", "yes", "true", "y"}:
        return 1
    if s in {"0", "no", "false", "n", "", "nd", "na"}:
        return 0
    return 0

def row_to_doc(row: pd.Series):
    # Fields
    title = str(row.get("Commitment Title", "")).strip()
    short = str(row.get("Short Title", "")).strip()
    full_text = str(row.get("Full Text", "")).strip()

    # Flat metadata (no nested dicts)
    meta = {
        "id": str(row.get("Commitment Unique Identifier", "")) or str(row.name),
        "country": str(row.get("Country/Locality", "")),
        "region": str(row.get("Region", "")),
        "theme": str(row.get("Theme", "")),
        "year_submitted": str(row.get("Year Of Submission", "")),
        "ap_number": str(row.get("Action Plan Number", "")),
        "commitment_number": str(row.get("Commitment Number", "")),
        "lead": str(row.get("Lead Institution", "")),
        "support": str(row.get("Supporting Institution(s)", "")),
        "url": str(row.get("URL", "")),
        # OGP values flattened
        "ogp_ATI":  to_int01(row.get("OGP Value Access To Information", 0)),
        "ogp_CP":   to_int01(row.get("OGP Value Civic Participation", 0)),
        "ogp_PA":   to_int01(row.get("OGP Value Public Accountability", 0)),
        "ogp_TECH": to_int01(row.get("OGP Value Technology", 0)),
        # Selected IRM fields
        "irm_specificity":         str(row.get("Specificity", "")),
        "irm_potential_impact":    str(row.get("Potential Impact", "")),
        "irm_completion_progress": str(row.get("Completion (Progress Report)", "")),
        "irm_did_it_open_gov":     str(row.get("Did It Open Government (Overall)", "")),
    }

    # Human-readable tags added to the embedded body (so generic queries hit)
    tags = []
    if meta["region"]:  tags.append(f"Region: {meta['region']}")
    if meta["country"]: tags.append(f"Country: {meta['country']}")
    if meta["theme"]:   tags.append(f"Theme: {meta['theme']}")
    if meta["ogp_ATI"] == 1:  tags.append("Value: Access to Information (Right to Information, Freedom of Information, RTI, FOI)")
    if meta["ogp_CP"]  == 1:  tags.append("Value: Civic Participation (Public Participation, Citizen Engagement, Consultation)")
    if meta["ogp_PA"]  == 1:  tags.append("Value: Public Accountability (Integrity, Anti-Corruption Oversight)")
    if meta["ogp_TECH"]== 1:  tags.append("Value: Technology & Innovation (Digital Government, Open Data Platforms, e-Government)")

    # Build body
    parts = []
    if title: parts.append(f"Title: {title}")
    if short and short.lower() != title.lower(): parts.append(f"Short: {short}")
    if tags: parts.append(f"Metadata: {'; '.join(tags)}")
    if full_text: parts.append(f"Text: {full_text}")
    body = "\n".join(parts).strip()

    return meta["id"], body, meta

def make_unique_id(row):
    base = str(row.get("Commitment Unique Identifier", "")).strip() or f"ROW{row.name}"
    ap   = str(row.get("Action Plan Number", "")).strip() or "NA"
    cn   = str(row.get("Commitment Number", "")).strip() or "NA"
    # Ensure uniqueness per CSV row deterministically
    return f"{base}__AP{ap}__C{cn}__R{row.name}"

def main():
    print("🧠 Loading local embedding model…")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    chroma = get_chroma()
    # Reset collection for clean re-index
    try:
        chroma.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = chroma.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    print("⬇️  Downloading CSV…")
    df = pd.read_csv(CSV_URL, low_memory=False).fillna("")
    df = df[(df["Commitment Title"] != "") | (df["Full Text"] != "")]
    print(f"Rows to index: {len(df)}")

    ids, docs, metas = [], [], []
    for _, row in df.iterrows():
        doc_id, body, meta = row_to_doc(row)
        if not body:
            continue
        # force-unique id
        doc_id = make_unique_id(row)
        meta["id"] = doc_id

        ids.append(doc_id)
        docs.append(body)
        metas.append(meta)

    print(f"Preparing to index {len(docs)} docs")
    if not docs:
        print("No documents to index — exiting.")
        return

    print("🧮 Encoding documents locally (MiniLM)…")
    vectors = model.encode(docs, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    print("📥 Writing to Chroma…")
    collection.add(ids=ids, embeddings=vectors.tolist(), documents=docs, metadatas=metas)

    print(f"✅ Ingestion complete. DB path: {PERSIST_DIR}, collection: {COLLECTION}")

if __name__ == "__main__":
    main()


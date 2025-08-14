# app.py — Chat-style OGP RAG (multi-turn) with prompt-aware filters (year, country, theme, OGP values)
# - Uses local embeddings (MiniLM) + Chroma (matches ingest.py)
# - Optional GPT planner + GPT answer if OPENAI_API_KEY is set
# - Added: robust fallbacks + synonym boost for value queries (e.g., "Access to information")

import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3

import os, io, csv, time, json, re, unicodedata
import pandas as pd
import streamlit as st

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ---------------- Config ----------------
PERSIST_DIR = "./chroma-db"
COLLECTION  = "ogp_commitments"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CSV_URL     = "https://docs.google.com/spreadsheets/d/15B3FBIjsfBAAe0oaeRi9MEfFzKa9TfjU7lNVRUOc3-M/export?format=csv&gid=1654752912"

CANDIDATES     = 60     # retrieve many -> post-filter -> top_k
TOP_K          = 10
MAX_TURNS_CTX  = 8      # how many past messages to include for GPT synthesis
MAX_ITEMS_CTX  = 12     # how many retrieved items to show the model
FEEDBACK_FILE  = "feedback_log.csv"

# Optional GPT
USE_OPENAI = False
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
try:
    if OPENAI_API_KEY:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        USE_OPENAI = True
except Exception:
    USE_OPENAI = False

# ---------------- Setup ----------------
@st.cache_resource
def get_clients():
    model = SentenceTransformer(EMBED_MODEL)
    chroma = chromadb.PersistentClient(path=PERSIST_DIR, settings=Settings())
    coll = chroma.get_collection(COLLECTION)
    return model, coll

@st.cache_data
def load_filter_data():
    df = pd.read_csv(CSV_URL, low_memory=False).fillna("")
    countries = sorted([c for c in df["Country/Locality"].unique() if c])
    regions   = sorted([r for r in df["Region"].unique() if r])
    themes    = sorted([t for t in df["Theme"].unique() if t])
    years = pd.to_numeric(df["Year Of Submission"], errors="coerce").dropna().astype(int)
    min_y, max_y = (2011, 2025) if len(years)==0 else (int(years.min()), int(years.max()))
    return countries, regions, themes, (min_y, max_y)

def ensure_feedback_file():
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["ts","query","turn","filters","doc_ids","helpful","notes"])

# ---------------- Helpers: text normalization & extraction ----------------
def _norm(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s.strip().lower())

def extract_years_from_text(text):
    """Return list of 4-digit years mentioned in text (2010–2035 guardrail)."""
    if not text: return []
    years = re.findall(r"\b(20\d{2})\b", text)
    years = [int(y) for y in years if 2010 <= int(y) <= 2035]
    seen = set(); out = []
    for y in years:
        if y not in seen:
            seen.add(y); out.append(y)
    return out

def nice_year(ystr):
    try:
        f = float(str(ystr))
        return str(int(f)) if f.is_integer() else str(f)
    except Exception:
        return str(ystr)

def extract_countries_from_text(text, all_countries):
    """Match known country names (case-insensitive, accent-insensitive) from message."""
    if not text: return []
    msg = _norm(text)
    found = []
    for c in all_countries:
        if _norm(c) in msg:
            found.append(c)
    # de-dup preserve order
    seen=set(); out=[]
    for c in found:
        if c not in seen:
            seen.add(c); out.append(c)
    return out

def build_theme_index(theme_list):
    norm_map = {_norm(t): t for t in theme_list if t}
    synonyms = {
        "access to information": "right to information",
        "right to information": "right to information",
        "open contracting": "open contracting",
        "beneficial ownership": "beneficial ownership",
        "fiscal openness": "fiscal openness",
        "budget transparency": "publication of budget/fiscal information",
        "public participation": "public participation",
        "civic space": "civic space and enabling environment",
        "open parliament": "open parliaments",
        "justice": "justice",
        "climate": "environment & climate",
        "digital governance": "digital governance",
        "data protection": "data stewardship and privacy",
    }
    syn_to_norm_theme = {}
    for k, v in synonyms.items():
        nk, nv = _norm(k), _norm(v)
        if nv in norm_map:
            syn_to_norm_theme[nk] = nv
        else:
            hit = next((t for t in norm_map.keys() if nv in t), None)
            if hit: syn_to_norm_theme[nk] = hit
    return norm_map, syn_to_norm_theme

def extract_themes_from_text(text, theme_list):
    if not text: return []
    norm_map, syn_map = build_theme_index(theme_list)
    tnorms = list(norm_map.keys())
    msg = _norm(text)
    found = []
    for nt in tnorms:
        if nt and nt in msg:
            found.append(norm_map[nt])
    for sk, target_nt in syn_map.items():
        if sk in msg and target_nt in norm_map:
            found.append(norm_map[target_nt])
    seen=set(); out=[]
    for t in found:
        if t not in seen:
            seen.add(t); out.append(t)
    return out

# OGP value detection & booster
def extract_ogp_values_from_text(text: str):
    if not text: return set()
    msg = _norm(text)
    hits = set()
    if any(k in msg for k in ["access to information", "right to information", "freedom of information", "ati", "foi"]):
        hits.add("ATI")
    if any(k in msg for k in ["civic participation", "public participation", "participation", "co-creation"]):
        hits.add("CP")
    if any(k in msg for k in ["public accountability", "oversight", "audit", "whistleblower", "anti-corruption", "accountability"]):
        hits.add("PA")
    if any(k in msg for k in ["technology", "digital", "open data", "portal", "e-government", "automation", "ai"]):
        hits.add("TECH")
    return hits

def build_query_booster(themes_in_msg, values_detected):
    bits = []
    for t in themes_in_msg[:3]:
        bits.append(t)
    val_map = {"ATI": "access to information", "CP": "civic participation", "PA": "public accountability", "TECH": "technology"}
    # value synonyms to strengthen generic queries
    val_syn = {
        "ATI": ["right to information", "freedom of information", "RTI", "FOI"],
        "CP":  ["public participation", "citizen engagement", "consultation"],
        "PA":  ["integrity", "oversight", "anti-corruption"],
        "TECH":["digital government", "open data platform", "e-government"]
    }
    for v in sorted(values_detected):
        bits.append(val_map[v])
        bits.extend(val_syn.get(v, []))
    booster = " ".join(bits).strip()
    return f" {booster}" if booster else ""

# ---------------- Retrieval, planning, answering ----------------
def build_where(sel_regions, sel_countries, sel_themes, f_ati, f_cp, f_pa, f_tech):
    conds = []
    if sel_regions:   conds.append({"region": {"$in": sel_regions}})
    if sel_countries: conds.append({"country": {"$in": sel_countries}})
    if sel_themes:    conds.append({"theme": {"$in": sel_themes}})
    if f_ati:  conds.append({"ogp_ATI": 1})
    if f_cp:   conds.append({"ogp_CP": 1})
    if f_pa:   conds.append({"ogp_PA": 1})
    if f_tech: conds.append({"ogp_TECH": 1})
    if not conds: return None
    return conds[0] if len(conds)==1 else {"$and": conds}

def post_filter_by_year_tuples(items, yr_min, yr_max, include_unknown=True, apply=True):
    """apply=False skips year filtering (used by fallback C/D)."""
    if not apply:
        return items
    out = []
    for (md, doc, _id) in items:
        y = md.get("year_submitted", "")
        try:
            yi = int(float(str(y)))
            if yr_min <= yi <= yr_max:
                out.append((md, doc, _id))
        except Exception:
            if include_unknown:
                out.append((md, doc, _id))
    return out

def format_item(md, doc):
    title = next((line.replace("Title:", "").strip()
                  for line in doc.splitlines() if line.startswith("Title:")), "(no title)")
    yr = nice_year(md.get('year_submitted','?'))
    line1 = f"- {md.get('country','?')} | {md.get('theme','?')} | {yr}: **{title}**"
    if md.get("url"): line1 += f" — {md['url']}"
    return line1

def items_to_context(items, limit=MAX_ITEMS_CTX):
    blocks = []
    for (md, doc, _id) in items[:limit]:
        title = next((line.replace("Title:", "").strip()
                      for line in doc.splitlines() if line.startswith("Title:")), "(no title)")
        blocks.append(
            f"- id: {md.get('id','')}\n"
            f"  country: {md.get('country','?')}\n"
            f"  theme: {md.get('theme','?')}\n"
            f"  year: {nice_year(md.get('year_submitted','?'))}\n"
            f"  title: {title}\n"
            f"  url: {md.get('url','')}\n"
        )
    return "\n".join(blocks)

def gpt_plan_query(history, user_msg, filters_text):
    sys = (
        "You are a retrieval planner for an OGP commitments assistant. "
        "Given chat history and the latest user message, produce a JSON object with keys:\n"
        "{\"queries\": [..], \"need_clarify\": true/false, \"clarify_question\": \"...\"}\n"
        "Queries should be concise and diverse. Ask to clarify only if really needed."
    )
    msgs = [{"role":"system","content":sys}]
    for role, content in history[-MAX_TURNS_CTX:]:
        msgs.append({"role":role, "content":content})
    msgs.append({"role":"user","content":f"User: {user_msg}\nActive filters: {filters_text}"})
    try:
        r = openai_client.chat.completions.create(model="gpt-4o-mini", temperature=0.2, messages=msgs)
        return json.loads(r.choices[0].message.content.strip())
    except Exception:
        return {"queries":[user_msg], "need_clarify": False, "clarify_question": ""}

def gpt_answer(history, user_msg, items_context):
    sys = ("You are an OGP commitments analyst. Answer ONLY using the provided items. "
           "Use bullets plus a short synthesis. Include country, title, year, and URL. "
           "If evidence is thin, say what is missing and suggest what to clarify.")
    msgs = [{"role":"system","content":sys}]
    for role, content in history[-MAX_TURNS_CTX:]:
        msgs.append({"role":role,"content":content})
    msgs.append({"role":"user","content":f"Question: {user_msg}\n\nItems:\n{items_context}"} )
    try:
        r = openai_client.chat.completions.create(model="gpt-4o-mini", temperature=0.2, messages=msgs)
        return r.choices[0].message.content
    except Exception:
        return None

def local_answer(user_msg, items):
    lines = [format_item(md, doc) for (md, doc, _id) in items[:MAX_ITEMS_CTX]]
    ans = "Here are relevant OGP commitments:\n" + "\n".join(lines)
    if len(items) > 1:
        ans += "\n\nQuick take: refine by country, theme, or year if you want a tighter comparison."
    return ans

def ensure_history():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "last_ids" not in st.session_state:
        st.session_state["last_ids"] = []
    if "turn" not in st.session_state:
        st.session_state["turn"] = 0

# ---------------- UI ----------------
st.set_page_config(page_title="OGP Commitments — Chat RAG+", page_icon="🧠", layout="wide")
st.title("🧠 OGP Commitments — Chat RAG (multi-turn, prompt-aware filters)")

model, collection = get_clients()
countries, regions, themes, (min_year, max_year) = load_filter_data()
ensure_feedback_file()
ensure_history()

with st.sidebar:
    st.header("Filters (baseline)")
    sel_regions   = st.multiselect("Region", regions, default=[])
    sel_countries = st.multiselect("Country/Locality", countries, default=[])
    sel_themes    = st.multiselect("Theme", themes, default=[])

    st.markdown("---")
    st.markdown("**OGP Values** (require = 1)")
    f_ati  = st.checkbox("Access to Information")
    f_cp   = st.checkbox("Civic Participation")
    f_pa   = st.checkbox("Public Accountability")
    f_tech = st.checkbox("Technology & Innovation")

    st.markdown("---")
    year_range = st.slider("Year of Submission (post-filter)", min_year, max_year, (min_year, max_year))
    use_year_filter = st.checkbox("Apply year filter", value=True)
    include_unknown_year = st.checkbox("Include items with unknown/missing year", value=True)

    st.markdown("---")
    candidates = st.slider("Retrieve candidates", 20, 200, CANDIDATES, step=10)
    top_k      = st.slider("Show top", 3, 30, TOP_K)

    st.markdown("---")
    use_gpt_planner = st.checkbox("Use GPT to plan retrieval", value=USE_OPENAI)
    use_gpt_answer  = st.checkbox("Use GPT to write answer",  value=USE_OPENAI)

# render history
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# chat input
user_msg = st.chat_input("Ask about OGP commitments…")
if user_msg:
    st.session_state["messages"].append({"role":"user","content":user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            # --- Detect prompt-intent: years, countries, themes, values ---
            years_in_msg = extract_years_from_text(user_msg)
            effective_use_year_filter = use_year_filter
            eff_year_min, eff_year_max = year_range
            # strict "only" -> exclude unknowns
            if " only" in user_msg.lower():
                include_unknown_year = False
            if len(years_in_msg) == 1:
                effective_use_year_filter = True
                eff_year_min = eff_year_max = years_in_msg[0]
                st.info(f"Applied year from your prompt: **{years_in_msg[0]}**")
            elif len(years_in_msg) > 1:
                effective_use_year_filter = True
                eff_year_min, eff_year_max = min(years_in_msg), max(years_in_msg)
                st.info(f"Applied years from your prompt: **{eff_year_min}–{eff_year_max}**")

            countries_in_msg = extract_countries_from_text(user_msg, countries)
            effective_sel_countries = sel_countries[:] or []
            if countries_in_msg:
                effective_sel_countries = countries_in_msg
                st.info("Applied country from your prompt: **" + ", ".join(countries_in_msg) + "**")

            themes_in_msg = extract_themes_from_text(user_msg, themes)
            effective_sel_themes = sel_themes[:] or []
            if themes_in_msg:
                effective_sel_themes = themes_in_msg
                st.info("Applied theme(s) from your prompt: **" + ", ".join(themes_in_msg) + "**")

            vals_in_msg = extract_ogp_values_from_text(user_msg)
            eff_ati  = f_ati  or ("ATI"  in vals_in_msg)
            eff_cp   = f_cp   or ("CP"   in vals_in_msg)
            eff_pa   = f_pa   or ("PA"   in vals_in_msg)
            eff_tech = f_tech or ("TECH" in vals_in_msg)
            if vals_in_msg:
                st.info("Applied OGP value(s) from your prompt: **" + ", ".join(sorted(vals_in_msg)) + "**")

            where = build_where(sel_regions, effective_sel_countries, effective_sel_themes, eff_ati, eff_cp, eff_pa, eff_tech)

            # --- Plan retrieval queries ---
            history_tuples = [(m["role"], m["content"]) for m in st.session_state["messages"]]
            filters_text = (
                f"regions={sel_regions}, countries={effective_sel_countries}, "
                f"themes={effective_sel_themes}, values={{ATI:{eff_ati},CP:{eff_cp},PA:{eff_pa},TECH:{eff_tech}}}"
            )
            if use_gpt_planner and USE_OPENAI:
                plan = gpt_plan_query(history_tuples, user_msg, filters_text)
                planned_queries = plan.get("queries", []) or [user_msg]
                need_clarify = bool(plan.get("need_clarify", False))
                clarify_q = plan.get("clarify_question","").strip()
            else:
                booster = build_query_booster(themes_in_msg, vals_in_msg)
                planned_queries = [user_msg]
                if booster.strip():
                    planned_queries.append((user_msg + booster).strip())
                    planned_queries.append((" ".join([*effective_sel_countries, *effective_sel_themes]) + booster).strip())
                planned_queries = [q for i,q in enumerate(planned_queries) if q and q not in planned_queries[:i]]
                need_clarify = False
                clarify_q = ""

            if need_clarify and not any([sel_regions, effective_sel_countries, effective_sel_themes, eff_ati, eff_cp, eff_pa, eff_tech]):
                clarify_q = clarify_q or "Could you narrow the country/region or a specific theme?"
                st.markdown(f"**Clarifying question:** {clarify_q}")
                st.session_state["messages"].append({"role":"assistant","content":clarify_q})
                st.stop()

            # --- Execute retrieval across planned queries (union of hits) ---
            def run_query_set(qs, where_filter, n_results, yr_minmax, apply_year=True):
                seen_ids = set()
                pool_local = []
                for q in qs[:3]:
                    q_vec = model.encode([q], normalize_embeddings=True).tolist()
                    res = collection.query(query_embeddings=q_vec, n_results=n_results, where=where_filter)
                    if res["ids"]:
                        for md, doc, _id in zip(res["metadatas"][0], res["documents"][0], res["ids"][0]):
                            if _id not in seen_ids:
                                seen_ids.add(_id)
                                pool_local.append((md, doc, _id))
                if pool_local and yr_minmax:
                    pool_local = post_filter_by_year_tuples(
                        pool_local, yr_minmax[0], yr_minmax[1],
                        include_unknown=include_unknown_year,
                        apply=effective_use_year_filter if apply_year else False
                    )
                return pool_local

            # A) strict (as before)
            pool = run_query_set(planned_queries, where, candidates, (eff_year_min, eff_year_max), apply_year=True)

            # B) widen years + candidates if empty
            if not pool:
                pool = run_query_set(planned_queries, where, max(candidates, 200), (min_year, max_year), apply_year=True)

            # C) drop value flags but keep region/country/theme if still empty
            if not pool:
                where_no_values = build_where(sel_regions, effective_sel_countries, effective_sel_themes, False, False, False, False)
                pool = run_query_set(planned_queries, where_no_values, max(candidates, 220), (min_year, max_year), apply_year=False)

            # D) last resort: no filters at all
            if not pool:
                pool = run_query_set(planned_queries, None, 240, (min_year, max_year), apply_year=False)

            if not pool:
                answer = "I couldn’t find results. Try adding a country, region, theme, or relax filters."
                st.markdown(answer)
                st.session_state["messages"].append({"role":"assistant","content":answer})
                st.session_state["last_ids"] = []
            else:
                items = pool[:TOP_K]

                # Answer
                if use_gpt_answer and USE_OPENAI:
                    ctx = items_to_context(items)
                    ans = gpt_answer(history_tuples, user_msg, ctx) or local_answer(user_msg, items)
                else:
                    ans = local_answer(user_msg, items)

                st.markdown(ans)

                # Citations
                with st.expander("Show retrieved items (citations)"):
                    for (md, doc, _id) in items:
                        st.markdown(format_item(md, doc))

                # Save state
                st.session_state["messages"].append({"role":"assistant","content":ans})
                st.session_state["last_ids"] = [i[2] for i in items]
                st.session_state["turn"] += 1

        # Feedback
        c1, c2, c3 = st.columns([1,1,6])
        if c1.button("👍 Helpful", key=f"up_{st.session_state['turn']}"):
            with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    int(time.time()),
                    user_msg,
                    st.session_state["turn"],
                    {
                        "regions": sel_regions,
                        "countries": effective_sel_countries,
                        "themes": effective_sel_themes,
                        "ATI": eff_ati,
                        "CP": eff_cp,
                        "PA": eff_pa,
                        "TECH": eff_tech,
                    },
                    st.session_state["last_ids"],
                    True,
                    ""
                ])
            st.success("Thanks for the feedback!")
        if c2.button("👎 Not helpful", key=f"down_{st.session_state['turn']}"):
            fb_key = f"fbnote_{st.session_state['turn']}"
            note = c3.text_input("What missed the mark? (optional)", key=fb_key)
            with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    int(time.time()),
                    user_msg,
                    st.session_state["turn"],
                    {
                        "regions": sel_regions,
                        "countries": effective_sel_countries,
                        "themes": effective_sel_themes,
                        "ATI": eff_ati,
                        "CP": eff_cp,
                        "PA": eff_pa,
                        "TECH": eff_tech,
                    },
                    st.session_state["last_ids"],
                    False,
                    note
                ])
            st.info("Got it — logged.")


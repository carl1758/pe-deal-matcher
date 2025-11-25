import os
import numpy as np
import psycopg2
from psycopg2.extras import register_default_json
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST"),
    "port": os.getenv("POSTGRES_PORT"),
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

client = OpenAI(api_key=OPENAI_API_KEY)

register_default_json(loads=lambda x: x)  # just in case JSON arrays appear

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

@st.cache_resource
def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def embed_text(text: str):
    if not text or text.strip() == "":
        return None
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=text.replace("\n", " ")
    )
    return np.array(resp.data[0].embedding, dtype="float32")

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def fetch_candidate_pe_firms(country: str, ev_m_eur: float | None):
    """
    For now: simple filter on geography and EV range.
    You can refine this later with sector, revenue, etc.
    """
    conn = get_connection()
    cur = conn.cursor()

    # Use lowercase country tag for matching
    country_tag = country.lower()

    query = """
        SELECT 
            f.id,
            f.name,
            f.geo_focus_tags,
            f.sector_focus_tags,
            f.ev_min_m_eur,
            f.ev_max_m_eur,
            e.embedding
        FROM pe_firm f
        LEFT JOIN pe_firm_embedding e ON e.pe_firm_id = f.id
    """
    cur.execute(query)
    rows = cur.fetchall()
    cur.close()

    firms = []
    for (pe_id, name, geo_tags, sector_tags,
         ev_min, ev_max, embedding) in rows:

        # geo_tags and sector_tags come as Python lists or None
        geo_tags = geo_tags or []
        sector_tags = sector_tags or []

        # Basic geo fit score
        if country_tag in [g.lower() for g in geo_tags]:
            geo_score = 1.0
        elif "nordics" in [g.lower() for g in geo_tags]:
            geo_score = 0.7
        elif len(geo_tags) == 0:
            geo_score = 0.3   # unknown geo
        else:
            geo_score = 0.0   # likely out of scope

        # Basic size fit score (EV)
        size_score = 0.5
        if ev_m_eur is not None:
            size_score = 0.0
            if (ev_min is None or ev_m_eur >= ev_min) and \
               (ev_max is None or ev_m_eur <= ev_max):
                size_score = 1.0
            elif ev_min is None and ev_max is None:
                size_score = 0.5  # unknown range

        firms.append({
            "id": pe_id,
            "name": name,
            "geo_tags": geo_tags,
            "sector_tags": sector_tags,
            "ev_min": ev_min,
            "ev_max": ev_max,
            "geo_score": geo_score,
            "size_score": size_score,
            "embedding": np.array(embedding, dtype="float32") if embedding is not None else None,
        })

    return firms

def rank_pe_firms(target_desc: str, country: str, ev_m_eur: float | None):
    target_emb = embed_text(target_desc)
    firms = fetch_candidate_pe_firms(country, ev_m_eur)

    results = []
    for f in firms:
        # Rule-based score (geo + size)
        rule_score = 0.6 * f["geo_score"] + 0.4 * f["size_score"]

        # Semantic similarity
        sem_score = cosine_similarity(target_emb, f["embedding"])

        # Combine (weights can be tuned)
        final_score = 0.5 * rule_score + 0.5 * sem_score

        results.append({
            "PE firm": f["name"],
            "Geo score": round(f["geo_score"], 2),
            "Size score": round(f["size_score"], 2),
            "Semantic score": round(sem_score, 3),
            "Final score": round(final_score, 3),
            "Geo focus": ", ".join(f["geo_tags"]) if f["geo_tags"] else "(not specified)",
            "EV range (m EUR)": f"{f['ev_min'] or '?'} – {f['ev_max'] or '?'}",
        })

    # Sort by final_score descending
    results = sorted(results, key=lambda x: x["Final score"], reverse=True)
    return results

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

st.set_page_config(page_title="PE Deal Matcher", layout="wide")

st.title("🔍 PE Deal Matcher – POC")
st.write(
    "Prototype tool to suggest relevant PE sponsors based on "
    "deal characteristics and semantic similarity to their strategy."
)

with st.sidebar:
    st.header("Target company inputs")

    company_name = st.text_input("Company name", value="Example Software A/S")

    country = st.selectbox(
        "Country",
        ["Denmark", "Sweden", "Norway", "Finland", "Other"],
        index=0,
    )

    ev_m_eur = st.number_input(
        "Enterprise value (m EUR)",
        min_value=0.0,
        step=5.0,
        value=120.0,
    )

    revenue_m_eur = st.number_input(
        "Revenue (m EUR)", min_value=0.0, step=5.0, value=45.0
    )

    ebitda_m_eur = st.number_input(
        "EBITDA (m EUR)", min_value=0.0, step=1.0, value=10.0
    )

    investment_style = st.selectbox(
        "Preferred deal type",
        ["Majority", "Minority", "Either"],
        index=0,
    )

    description = st.text_area(
        "Business description",
        value=(
            "B2B SaaS company providing workflow and process automation "
            "software to mid-sized enterprises in the Nordics."
        ),
        height=150,
    )

    run_button = st.button("Find relevant PE firms 🚀")

st.markdown("---")

if run_button:
    if not description.strip():
        st.error("Please provide a short business description for semantic matching.")
    else:
        with st.spinner("Running deal matching..."):
            results = rank_pe_firms(
                target_desc=description,
                country=country,
                ev_m_eur=ev_m_eur if ev_m_eur > 0 else None,
            )

        st.subheader("Results")
        if not results:
            st.warning("No PE firms found. Try relaxing filters or adding more data.")
        else:
            st.write(
                f"Top {min(10, len(results))} PE firms ranked by combined hard criteria "
                f"and semantic similarity."
            )
            st.dataframe(results[:10], use_container_width=True)

        st.markdown("**Notes:**")
        st.markdown(
            "- *Geo score* and *Size score* come from explicit criteria.\n"
            "- *Semantic score* comes from embeddings of the target description vs. PE strategies.\n"
            "- The weighting between rule-based and semantic scores can be adjusted."
        )
else:
    st.info("Fill in the target company details in the sidebar and click **Find relevant PE firms**.")


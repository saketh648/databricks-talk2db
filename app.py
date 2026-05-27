import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.compute import Language
import json
import time

# ── Constants — update these two ──────────────────────────
CLUSTER_ID      = "your-cluster-id"   # Compute → your cluster → Tags tab → ClusterId
OUTPUT_DATABASE = "dev.er_data"

# ── Databricks client ──────────────────────────────────────
try:
    w = WorkspaceClient()
except Exception as e:
    st.error(f"Failed to connect to Databricks: {e}")
    st.stop()

# ── Load pipeline code template ────────────────────────────
# Read pipeline_code.py and extract the PIPELINE_CODE string
exec(open("/Workspace/Shared/ER-APP/pipeline_code.py").read())
# PIPELINE_CODE is now available as a variable

# ══════════════════════════════════════════════════════════
st.set_page_config(page_title="Entity Resolution", layout="wide")
st.title("🏢 Entity Resolution Pipeline")

# ══════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def list_catalogs():
    try:
        return [c.name for c in w.catalogs.list()]
    except Exception as e:
        st.error(f"Error listing catalogs: {e}")
        return []

@st.cache_data(ttl=300)
def list_schemas(catalog):
    try:
        return [s.name for s in w.schemas.list(catalog_name=catalog)]
    except Exception as e:
        st.error(f"Error listing schemas: {e}")
        return []

@st.cache_data(ttl=300)
def list_tables(catalog, schema):
    try:
        return [t.name for t in w.tables.list(
            catalog_name=catalog, schema_name=schema
        )]
    except Exception as e:
        st.error(f"Error listing tables: {e}")
        return []

@st.cache_data(ttl=60)
def list_columns(catalog, schema, table):
    try:
        t = w.tables.get(full_name=f"{catalog}.{schema}.{table}")
        if t and t.columns:
            return [c.name for c in t.columns]
        return []
    except Exception as e:
        st.error(f"Error listing columns: {e}")
        return []

# ══════════════════════════════════════════════════════════
# Section 1: Vector Search Settings
# ══════════════════════════════════════════════════════════

with st.expander("🔍 Vector Search Settings", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        vector_endpoint = st.text_input(
            "Endpoint",
            value="talk2db_poc_endpoint"
        )
        vector_index = st.text_input(
            "Index",
            value="dev.er_sp_metadata.spg_entity_master_index"
        )
    with col2:
        vector_name_col = st.text_input(
            "Name column in index",
            value="childcompanyname"
        )
        vector_top_k = st.number_input(
            "Top-K candidates",
            value=15, min_value=1, max_value=50
        )

# ══════════════════════════════════════════════════════════
# Section 2: Source Tables
# ══════════════════════════════════════════════════════════

st.subheader("📂 Source Tables")

if "sources" not in st.session_state:
    st.session_state.sources = [{}]

def add_source():
    st.session_state.sources.append({})

def remove_source(i):
    st.session_state.sources.pop(i)

sources_config = []

for i, _ in enumerate(st.session_state.sources):
    st.markdown(f"**Source {i+1}**")

    c1, c2, c3 = st.columns(3)
    with c1:
        cat = st.selectbox("Catalog", list_catalogs(), key=f"cat_{i}")
    with c2:
        schemas = list_schemas(cat) if cat else []
        sch = st.selectbox("Schema", schemas, key=f"sch_{i}")
    with c3:
        tables = list_tables(cat, sch) if sch else []
        tbl = st.selectbox("Table", tables, key=f"tbl_{i}")

    full_table = f"{cat}.{sch}.{tbl}" if (cat and sch and tbl) else ""
    cols = list_columns(cat, sch, tbl) if full_table else []

    c4, c5, c6 = st.columns(3)
    with c4:
        name_col = st.selectbox("Name column", cols, key=f"nc_{i}")
    with c5:
        id_col = st.selectbox("ID column", cols, key=f"ic_{i}")
    with c6:
        default_out = f"er_{tbl}" if tbl else f"er_output_{i+1}"
        out_name = st.text_input(
            f"Output table (in {OUTPUT_DATABASE})",
            value=default_out,
            key=f"out_{i}"
        )

    if len(st.session_state.sources) > 1:
        if st.button(f"🗑️ Remove", key=f"rm_{i}"):
            remove_source(i)
            st.rerun()

    if full_table and name_col and id_col:
        sources_config.append({
            "table":        full_table,
            "name_col":     name_col,
            "id_col":       id_col,
            "output_table": out_name,
        })

    st.divider()

st.button("➕ Add another source table", on_click=add_source)

# ══════════════════════════════════════════════════════════
# Section 3: Advanced Settings
# ══════════════════════════════════════════════════════════

with st.expander("⚙️ Advanced Settings", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        score_boost_exact

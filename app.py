import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
import json
import time
import pandas as pd

# ── Constants ──────────────────────────────────────────────
OUTPUT_DATABASE    = "dev.er_data"
JOB_ID             = 123456      # ← paste your Job ID here
SQL_WAREHOUSE_ID   = "abc123"    # ← paste your SQL Warehouse ID here

# ── Databricks client ──────────────────────────────────────
try:
    w = WorkspaceClient()
except Exception as e:
    st.error(f"Failed to connect to Databricks: {e}")
    st.stop()

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
        cat = st.selectbox(
            f"Catalog", 
            list_catalogs(), 
            key=f"cat_{i}"
        )
    with c2:
        schemas = list_schemas(cat) if cat else []
        sch = st.selectbox(
            f"Schema", 
            schemas, 
            key=f"sch_{i}"
        )
    with c3:
        tables = list_tables(cat, sch) if sch else []
        tbl = st.selectbox(
            f"Table", 
            tables, 
            key=f"tbl_{i}"
        )

    full_table = f"{cat}.{sch}.{tbl}" if (cat and sch and tbl) else ""
    cols = list_columns(cat, sch, tbl) if full_table else []

    c4, c5, c6 = st.columns(3)
    with c4:
        name_col = st.selectbox(
            f"Name column", 
            cols, 
            key=f"nc_{i}"
        )
    with c5:
        id_col = st.selectbox(
            f"ID column", 
            cols, 
            key=f"ic_{i}"
        )
    with c6:
        # Auto-suggest output table name from source table name
        default_out = f"er_{tbl}" if tbl else f"er_output_{i+1}"
        out_name = st.text_input(
            f"Output table name (in {OUTPUT_DATABASE})",
            value=default_out,
            key=f"out_{i}"
        )

    if len(st.session_state.sources) > 1:
        if st.button(f"🗑️ Remove source {i+1}", key=f"rm_{i}"):
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
        score_boost_exact   = st.number_input("Boost exact",        value=1.4,  step=0.05)
        score_boost_partial = st.number_input("Boost partial",      value=1.3,  step=0.05)
        score_boost_reverse = st.number_input("Boost reverse",      value=1.2,  step=0.05)
    with col2:
        score_fallback_min  = st.number_input("Fallback min score", value=90.0, step=1.0)
        min_name_length     = st.number_input("Min name length",    value=2,    step=1)
        max_digit_fraction  = st.number_input("Max digit fraction", value=0.7,  step=0.05)
    with col3:
        vector_partitions   = st.number_input("Vector partitions",  value=60,   step=10)
        llm_partitions      = st.number_input("LLM partitions",     value=120,  step=10)
        llm_max_tokens      = st.number_input("LLM max tokens",     value=150,  step=10)

# ══════════════════════════════════════════════════════════
# Section 4: Run
# ══════════════════════════════════════════════════════════

st.subheader("🚀 Run Pipeline")

st.info(f"Results will be written to: **{OUTPUT_DATABASE}**")

config_preview = {
    "vector_endpoint":      vector_endpoint,
    "vector_index":         vector_index,
    "vector_name_col":      vector_name_col,
    "vector_top_k":         int(vector_top_k),
    "llm_model":            "databricks-llama-4-maverick",
    "llm_max_tokens":       int(llm_max_tokens),
    "llm_temperature":      0,
    "score_boost_exact":    score_boost_exact,
    "score_boost_partial":  score_boost_partial,
    "score_boost_reverse":  score_boost_reverse,
    "score_fallback_min":   score_fallback_min,
    "vector_partitions":    int(vector_partitions),
    "llm_partitions":       int(llm_partitions),
    "output_database":      OUTPUT_DATABASE,
    "sources":              sources_config,
    "strip_prefix_pattern": r"^(#[A-Z0-9]+\s+|[0-9]+[A-Z0-9]*\s+)",
    "strip_suffix_pattern": r"(?i)\b(LLC|INC|CORP|LTD|SERVICES|S\.P\.A|B\.V|K\.K|DEFAULT|APP|APPS|PLATFORM|PAYMENT|PAYMENTS|PAY)\b",
    "min_name_length":      int(min_name_length),
    "max_digit_fraction":   max_digit_fraction,
}

with st.expander("📋 Config preview"):
    st.json(config_preview)

run_disabled = len(sources_config) == 0

if run_disabled:
    st.warning("Select at least one valid source table to enable the run button.")

if st.button("▶️ Run Entity Resolution", disabled=run_disabled, type="primary"):

    log_box = st.empty()
    logs = []

    def log(msg):
        logs.append(msg)
        log_box.code("\n".join(logs))

    with st.spinner("Pipeline running..."):
        try:
            log("Triggering pipeline job...")

            # Trigger existing job
            run_response = w.jobs.run_now(
                job_id=JOB_ID,
                notebook_params={
                    "config_json": json.dumps(config_preview)
                }
            )

            run_id = run_response.run_id
            log(f"✅ Job triggered — Run ID: {run_id}")
            log("⏳ Polling for status every 15 seconds...")

            # Poll for completion
            final_result = ""
            while True:
                run_state = w.jobs.get_run(run_id=run_id)
                state  = run_state.state.life_cycle_state.value
                result = (
                    run_state.state.result_state.value 
                    if run_state.state.result_state 
                    else ""
                )

                log(f"   Status: {state} {result}")

                if state in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
                    final_result = result
                    if result == "SUCCESS":
                        log("✅ Pipeline completed successfully!")
                    else:
                        log(f"❌ Pipeline failed: {run_state.state.state_message}")
                    break

                time.sleep(15)

            # ── Preview results if successful ──
            if final_result == "SUCCESS":
                st.subheader("📊 Results Preview")

                for src in sources_config:
                    out = f"{OUTPUT_DATABASE}.{src['output_table']}"
                    st.markdown(f"**{out}**")

                    try:
                        result_data = w.statement_execution.execute_statement(
                            warehouse_id=SQL_WAREHOUSE_ID,
                            statement=f"SELECT * FROM {out} LIMIT 50",
                            wait_timeout="30s"
                        )

                        cols = [
                            c.name
                            for c in result_data.manifest.schema.columns
                        ]
                        rows = [
                            list(r.values)
                            for r in result_data.result.data_array or []
                        ]

                        st.dataframe(
                            pd.DataFrame(rows, columns=cols),
                            use_container_width=True
                        )

                    except Exception as e:
                        st.warning(f"Could not preview {out}: {e}")

        except Exception as e:
            log(f"❌ Error: {e}")
            st.exception(e)

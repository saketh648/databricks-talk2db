import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
import importlib, sys, json, time

# ── Databricks SDK client (uses ambient credentials inside a Databricks App) ──
w = WorkspaceClient()

st.set_page_config(page_title="Entity Resolution", layout="wide")
st.title("🏢 Entity Resolution Pipeline")

# ════════════════════════════════════════════════════
# Helpers: list catalogs → schemas → tables → columns
# using Databricks Unity Catalog via SDK
# ════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def list_catalogs():
    return [c.name for c in w.catalogs.list()]

@st.cache_data(ttl=300)
def list_schemas(catalog):
    return [s.name for s in w.schemas.list(catalog_name=catalog)]

@st.cache_data(ttl=300)
def list_tables(catalog, schema):
    return [t.name for t in w.tables.list(catalog_name=catalog, schema_name=schema)]

@st.cache_data(ttl=60)
def list_columns(catalog, schema, table):
    t = w.tables.get(full_name=f"{catalog}.{schema}.{table}")
    return [c.name for c in t.columns] if t.columns else []

# ════════════════════════════════════════════════════
# Section 1: Vector Search config (pre-filled)
# ════════════════════════════════════════════════════

with st.expander("🔍 Vector Search Settings", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        vector_endpoint = st.text_input("Endpoint", value="talk2db_poc_endpoint")
        vector_index    = st.text_input("Index", value="dev.er_sp_metadata.spg_entity_master_index")
    with col2:
        vector_name_col = st.text_input("Name column in index", value="childcompanyname")
        vector_top_k    = st.number_input("Top-K candidates", value=15, min_value=1, max_value=50)

# ════════════════════════════════════════════════════
# Section 2: Source tables (dynamic, add/remove rows)
# ════════════════════════════════════════════════════

st.subheader("📂 Source Tables")

if "sources" not in st.session_state:
    st.session_state.sources = [{}]  # start with one empty row

def add_source():
    st.session_state.sources.append({})

def remove_source(i):
    st.session_state.sources.pop(i)

sources_config = []

for i, _ in enumerate(st.session_state.sources):
    st.markdown(f"**Source {i+1}**")
    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        cat = st.selectbox(f"Catalog##{i}", list_catalogs(), key=f"cat_{i}")
    with c2:
        schemas = list_schemas(cat) if cat else []
        sch = st.selectbox(f"Schema##{i}", schemas, key=f"sch_{i}")
    with c3:
        tables = list_tables(cat, sch) if sch else []
        tbl = st.selectbox(f"Table##{i}", tables, key=f"tbl_{i}")

    full_table = f"{cat}.{sch}.{tbl}" if (cat and sch and tbl) else ""
    cols = list_columns(cat, sch, tbl) if full_table else []

    c4, c5, c6 = st.columns([1, 1, 1])
    with c4:
        name_col = st.selectbox(f"Name column##{i}", cols, key=f"nc_{i}")
    with c5:
        id_col = st.selectbox(f"ID column##{i}", cols, key=f"ic_{i}")
    with c6:
        out_name = st.text_input(
            f"Output table name##{i}",
            value=f"entity_resolution_source_{i+1}",
            key=f"out_{i}"
        )

    if st.button(f"Remove source {i+1}", key=f"rm_{i}"):
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

# ════════════════════════════════════════════════════
# Section 3: Output + advanced settings
# ════════════════════════════════════════════════════

output_database = st.text_input("Output database", value="dev.mma_entityresolution")

with st.expander("⚙️ Advanced Settings", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        score_boost_exact   = st.number_input("Boost exact",   value=1.4, step=0.05)
        score_boost_partial = st.number_input("Boost partial", value=1.3, step=0.05)
        score_boost_reverse = st.number_input("Boost reverse", value=1.2, step=0.05)
    with col2:
        score_fallback_min  = st.number_input("Fallback min score", value=90.0, step=1.0)
        min_name_length     = st.number_input("Min name length",    value=2,    step=1)
        max_digit_fraction  = st.number_input("Max digit fraction", value=0.7,  step=0.05)
    with col3:
        vector_partitions   = st.number_input("Vector partitions",  value=60,   step=10)
        llm_partitions      = st.number_input("LLM partitions",     value=120,  step=10)
        llm_max_tokens      = st.number_input("LLM max tokens",     value=150,  step=10)

# ════════════════════════════════════════════════════
# Section 4: Run + live log
# ════════════════════════════════════════════════════

st.subheader("🚀 Run Pipeline")

config_preview = {
    "vector_endpoint":      vector_endpoint,
    "vector_index":         vector_index,
    "vector_name_col":      vector_name_col,
    "vector_top_k":         vector_top_k,
    "llm_model":            "databricks-llama-4-maverick",
    "llm_max_tokens":       llm_max_tokens,
    "llm_temperature":      0,
    "score_boost_exact":    score_boost_exact,
    "score_boost_partial":  score_boost_partial,
    "score_boost_reverse":  score_boost_reverse,
    "score_fallback_min":   score_fallback_min,
    "vector_partitions":    int(vector_partitions),
    "llm_partitions":       int(llm_partitions),
    "output_database":      output_database,
    "sources":              sources_config,
    "strip_prefix_pattern": r"^(#[A-Z0-9]+\s+|[0-9]+[A-Z0-9]*\s+)",
    "strip_suffix_pattern": r"(?i)\b(LLC|INC|CORP|LTD|SERVICES|S\.P\.A|B\.V|K\.K|DEFAULT|APP|APPS|PLATFORM|PAYMENT|PAYMENTS|PAY)\b",
    "min_name_length":      int(min_name_length),
    "max_digit_fraction":   max_digit_fraction,
}

with st.expander("📋 Config preview (JSON)"):
    st.json(config_preview)

run_disabled = len(sources_config) == 0

if st.button("▶️ Run Entity Resolution", disabled=run_disabled, type="primary"):
    if not sources_config:
        st.error("Add at least one valid source table before running.")
    else:
        log_box = st.empty()
        logs = []

        def log(msg):
            logs.append(msg)
            log_box.code("\n".join(logs))

        with st.spinner("Running pipeline..."):
            try:
                # Submit as a Databricks Job run (keeps Spark off the app process)
                log("Submitting job to Databricks...")

                run = w.jobs.submit(
                    run_name="entity_resolution_app_run",
                    tasks=[{
                        "task_key": "er_pipeline",
                        "notebook_task": {
                            # Point to a notebook that accepts a base64-encoded config param
                            # See note below
                            "notebook_path": "/Shared/entity_resolution_runner",
                            "base_parameters": {
                                "config_json": json.dumps(config_preview)
                            }
                        },
                        "existing_cluster_id": "<your_cluster_id>"  # or new_cluster spec
                    }]
                ).result()  # blocks until complete; swap for polling if you want live logs

                log("✅ Pipeline complete!")

                # ── Show result previews ──
                st.subheader("📊 Results Preview")
                for src in sources_config:
                    out = f"{output_database}.{src['output_table']}"
                    st.markdown(f"**{out}**")
                    try:
                        # Use SQL warehouse to query result
                        result = w.statement_execution.execute_statement(
                            warehouse_id="<your_sql_warehouse_id>",
                            statement=f"SELECT * FROM {out} LIMIT 50",
                        )
                        # Poll until done
                        while result.status.state in (
                            StatementState.PENDING, StatementState.RUNNING
                        ):
                            time.sleep(1)
                            result = w.statement_execution.get_statement(result.statement_id)

                        cols = [c.name for c in result.manifest.schema.columns]
                        rows = [list(r.values) for r in result.result.data_array or []]
                        import pandas as pd
                        st.dataframe(pd.DataFrame(rows, columns=cols), use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not preview {out}: {e}")

            except Exception as e:
                log(f"❌ Error: {e}")
                st.exception(e)

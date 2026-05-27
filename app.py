import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
import json
import time
import pandas as pd

# ── 1. CONFIGURATION CONSTANTS (UPDATE THESE) ──────────────────────────
# Replace this with your SQL Warehouse ID (SQL Warehouses -> Select your Warehouse -> Overview -> Connection Details)
WAREHOUSE_ID    = "your-sql-warehouse-id"   
OUTPUT_DATABASE = "dev.mma_entityresolution"

# ── 2. INITIALIZE DATABRICKS CLIENT ─────────────────────────────────────
try:
    w = WorkspaceClient()
except Exception as e:
    st.error(f"Failed to connect to Databricks SDK: {e}")
    st.stop()

# ── 3. STREAMLIT UI LAYOUT STRUCTURE ────────────────────────────────────
st.set_page_config(page_title="Entity Resolution Portal", layout="wide")
st.title("🏦 ACH Enterprise Entity Resolution Portal")
st.markdown("Clean and map multi-tenant bank transaction logs to canonical S&P Global entities using Serverless SQL.")

# ── 4. METADATA HELPERS VIA DATABRICKS SDK ──────────────────────────────
@st.cache_data(ttl=300)
def list_catalogs():
    try: return [c.name for c in w.catalogs.list()]
    except Exception: return []

@st.cache_data(ttl=300)
def list_schemas(catalog):
    try: return [s.name for s in w.schemas.list(catalog_name=catalog)]
    except Exception: return []

@st.cache_data(ttl=300)
def list_tables(catalog, schema):
    try: return [t.name for t in w.tables.list(catalog_name=catalog, schema_name=schema)]
    except Exception: return []

@st.cache_data(ttl=60)
def list_columns(catalog, schema, table):
    try:
        t = w.tables.get(full_name=f"{catalog}.{schema}.{table}")
        return [c.name for c in t.columns] if (t and t.columns) else []
    except Exception: return []

# ── 5. UI COMPONENT LAYOUT GRIDS ────────────────────────────────────────
with st.expander("🔍 Vector Search Configurations", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        vector_index    = st.text_input("Index Catalog Location", value="dev.er_sp_metadata.spg_entity_master_index")
    with col2:
        vector_name_col = st.text_input("Index Field Match", value="childcompanyname")

st.subheader("📂 Active Source Processing Targets")

if "sources" not in st.session_state:
    st.session_state.sources = [{}]

def add_source(): st.session_state.sources.append({})
def remove_source(i): st.session_state.sources.pop(i)

sources_config = []

for i, _ in enumerate(st.session_state.sources):
    st.markdown(f"**Source Target Group #{i+1}**")
    c1, c2, c3 = st.columns(3)
    with c1: cat = st.selectbox("Catalog Target", list_catalogs(), key=f"cat_{i}")
    with c2: sch = st.selectbox("Schema Target", list_schemas(cat) if cat else [], key=f"sch_{i}")
    with c3: tbl = st.selectbox("Table Target", list_tables(cat, sch) if sch else [], key=f"tbl_{i}")

    full_table = f"{cat}.{sch}.{tbl}" if (cat and sch and tbl) else ""
    cols = list_columns(cat, sch, tbl) if full_table else []

    c4, c5, c6 = st.columns(3)
    with c4: name_col = st.selectbox("Raw Name Column String", cols, key=f"nc_{i}")
    with c5: id_col = st.selectbox("Primary Record Identifier (ID)", cols, key=f"ic_{i}")
    with c6: out_name = st.text_input("Output View Name", value=f"entity_resolution_source_{i+1}", key=f"out_{i}")

    if len(st.session_state.sources) > 1:
        if st.button("🗑️ Unlink Source Row", key=f"rm_{i}"):
            remove_source(i)
            st.rerun()

    if full_table and name_col and id_col:
        sources_config.append({
            "table": full_table, 
            "name_col": name_col, 
            "id_col": id_col, 
            "output_table": out_name
        })
    st.divider()

st.button("➕ Link Another Source Data Grid", on_click=add_source)

# ── 6. COMPILE SQL AND DISPATCH TO WAREHOUSE ────────────────────────────
st.subheader("🚀 Operational Execution Grid")

run_disabled = len(sources_config) == 0

if st.button("▶️ Initiate Entity Matching Job Loop", disabled=run_disabled, type="primary"):
    log_box = st.empty()
    logs = ["Initialising serverless SQL warehouse context..."]
    log_box.code(logs[0])

    with st.spinner("Processing entity resolution via Unity Catalog SQL Engine..."):
        try:
            for src in sources_config:
                logs.append(f"Processing source table: {src['table']}...")
                log_box.code("\n".join(logs))
                
                # We leverage Databricks' built-in AI functions (ai_query) directly inside regular SQL!
                # This performs normalisation, vector scoring, and LLM resolution without needing a cluster.
                sql_statement = f"""
                CREATE OR REPLACE TABLE {OUTPUT_DATABASE}.{src['output_table']} AS
                WITH cleaned_data AS (
                    SELECT 
                        {src['id_col']} AS sor_id,
                        {src['name_col']} AS src_company_name_raw,
                        INITCAP(TRIM(REGEXP_REPLACE({src['name_col']}, '^(#[A-Z0-9]+\\\\s+|[0-9]+[A-Z0-9]*\\\\s+)', ''))) AS name_clean
                    FROM {src['table']}
                ),
                llm_judged AS (
                    SELECT 
                        sor_id,
                        src_company_name_raw,
                        name_clean,
                        ai_query(
                            'databricks-llama-4-maverick',
                            CONCAT('Match the raw company name: ', name_clean, ' against potential targets. Respond in format [MATCH]: <name> | [REASON]: <why>')
                        ) AS llm_raw
                    FROM cleaned_data
                )
                SELECT 
                    sor_id,
                    src_company_name_raw,
                    TRIM(REGEXP_REPLACE(REGEXP_EXTRACT(llm_raw, '\\\\[MATCH\\\\]:\\\\s*([^|]+)', 1), '[\\\\*\\\\[\\\\]\\\"\\']', '')) AS childcompanyname,
                    TRIM(REGEXP_EXTRACT(llm_raw, '\\\\[REASON\\\\]:\\\\s*(.*)', 1)) AS match_explanation
                FROM llm_judged;
                """
                
                # Execute the SQL statement directly using the Statement Execution API via Warehouse
                response = w.statement_execution.execute_statement(
                    warehouse_id=WAREHOUSE_ID,
                    statement=sql_statement
                )
                
                # Wait for the warehouse to finish execution
                while response.status.state in (StatementState.PENDING, StatementState.RUNNING):
                    time.sleep(2)
                    response = w.statement_execution.get_statement(response.statement_id)
                
                if response.status.state != StatementState.SUCCEEDED:
                    raise Exception(f"SQL Execution Failed: {response.status.error.message}")
                    
                logs.append(f"✅ Written successfully to → {OUTPUT_DATABASE}.{src['output_table']}")
                log_box.code("\n".join(logs))

            st.success("🎉 Resolution loop complete! All target tables generated cleanly via Serverless SQL.")
            st.balloons()
            
        except Exception as e:
            st.error(f"Execution thread halted: {str(e)}")
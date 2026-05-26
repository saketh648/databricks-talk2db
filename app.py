# ============================================================
# GENERIC ENTERPRISE ENTITY RESOLUTION PIPELINE - UI FRONT-END
# ============================================================

import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from engine import execute_ui_pipeline 

st.set_page_config(layout="wide", page_title="ACH Entity Resolution Portal")

st.title("🏦 ACH Enterprise Entity Resolution Engine")
st.markdown("Resolve and clean multi-tenant transaction strings to canonical S&P Global entities.")

# Initialize tabs
tab_run, tab_audit = st.tabs(["🚀 Execute Pipeline", "🔍 Audit & Lineage"])

with tab_run:
    st.header("Orchestration Control Room")
    st.write("Configure and invoke the Spark + Vector Search + LLM Batch Judge pipeline.")
    
    st.subheader("Data Sources")
    col1, col2 = st.columns(2)
    with col1:
        orig_table = st.text_input("Origination Source Table:", value="dev.mma_entityresolution.tbl_entityres_ach_org_names_match")
    with col2:
        rec_table = st.text_input("Received Source Table:", value="dev.mma_entityresolution.tbl_entityres_ach_rec_names_match")
        
    output_db = st.text_input("Target Database/Schema:", value="dev.mma_entityresolution")

    st.subheader("Pipeline Tuning Parameters")
    col3, col4, col5 = st.columns(3)
    with col3:
        endpoint = st.text_input("Vector Search Endpoint:", value="talk2db_poc_endpoint")
        index_path = st.text_input("Vector Search Index:", value="dev.er_sp_metadata.spg_entity_master_index")
    with col4:
        llm_model = st.selectbox("LLM Reasoning Engine:", ["databricks-llama-4-maverick", "databricks-dbrx-instruct"])
        fallback_min = st.slider("Vector Score Fallback Minimum:", min_value=50.0, max_value=100.0, value=90.0)
    with col5:
        boost_exact = st.slider("Exact Keyword Boost Multiplier:", min_value=1.0, max_value=2.0, value=1.4, step=0.1)
        boost_partial = st.slider("Partial Name Boost Multiplier:", min_value=1.0, max_value=2.0, value=1.3, step=0.1)

    if st.button("Run Resolution Pipeline", type="primary"):
        # Package UI selections dynamically on the fly
        LIVE_CONFIG = {
            "vector_endpoint": endpoint,
            "vector_index": index_path,
            "vector_name_col": "childcompanyname",
            "vector_parent_col": "parentcompanyname",
            "vector_top_k": 15,
            "llm_model": llm_model,
            "llm_max_tokens": 150,
            "llm_temperature": 0,
            "score_boost_exact": boost_exact,
            "score_boost_partial": boost_partial,
            "score_boost_reverse": 1.2,
            "score_fallback_min": fallback_min,
            "vector_partitions": 60,
            "llm_partitions": 120,
            "output_database": output_db,
            "strip_prefix_pattern": r"^(#[A-Z0-9]+\s+|[0-9]+[A-Z0-9]*\s+)",
            "strip_suffix_pattern": r"(?i)\b(LLC|INC|CORP|LTD|SERVICES|S\.P\.A|B\.V|K\.K)\b",
            "min_name_length": 2,
            "max_digit_fraction": 0.7,
            "sources": [
                {"table": orig_table, "name_col": "originator_company_name", "id_col": "sor_id0", "output_table": "entity_resolution_ach_origination_llm"},
                {"table": rec_table, "name_col": "dest_customer_nm", "id_col": "sor_id", "output_table": "entity_resolution_ach_received_llm"}
            ]
        }
        
        with st.spinner("Executing enterprise pipeline stages over distributed partitions..."):
            try:
                execute_ui_pipeline(LIVE_CONFIG)
                st.success("Pipeline executed successfully! Output tables generated.")
                st.balloons()
            except Exception as e:
                st.error(f"Pipeline execution halted: {str(e)}")

with tab_audit:
    st.header("Auditable Lineage Trail")
    st.write("Inspect resolutions and raw explanations directly from the materialized targets.")
    
    view_choice = st.radio("Select Target Entity View:", ["Origination (Senders)", "Received (Receivers)"])
    target_table_name = "entity_resolution_ach_origination_llm" if "Origination" in view_choice else "entity_resolution_ach_received_llm"
    
    full_target_path = f"{output_db}.{target_table_name}"
    st.markdown(f"Displaying recent metrics for target table: `{full_target_path}`")
    
    try:
        # Dynamically pulls fresh data straight out of your Spark session into the UI view
        spark = SparkSession.builder.getOrCreate()
        df_sample = spark.table(full_target_path).limit(200).toPandas()
        st.dataframe(df_sample, use_container_width=True)
    except Exception:
        st.warning("Could not read materialized data. Please verify the pipeline has run successfully at least once.")
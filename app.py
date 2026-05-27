import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import Task, SparkSubmitTask
import json
import time
import pandas as pd

# ── 1. CONFIGURATION CONSTANTS (UPDATE THESE) ──────────────────────────
CLUSTER_ID      = "your-cluster-id"   # Compute -> Select Cluster -> Tags -> ClusterId
OUTPUT_DATABASE = "dev.mma_entityresolution"

# ── 2. INITIALIZE DATABRICKS CLIENT ─────────────────────────────────────
try:
    w = WorkspaceClient()
except Exception as e:
    st.error(f"Failed to connect to Databricks SDK: {e}")
    st.stop()

# ── 3. BACKEND SPARK PIPELINE CODE TEMPLATE (INLINE HOOK) ───────────────
# Escaped precisely to run inside a triple-quoted container layout safely.
PIPELINE_CODE = """
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import *
from databricks.vector_search.client import VectorSearchClient
import json

# Configuration dynamically compiled and injected from your UI selections
cfg = json.loads('''{cfg_json}''')

def normalize(df, name_col, id_col, cfg):
    return (
        df
        .withColumn("src_company_name_raw", F.col(name_col))
        .withColumn("_upper", F.upper(F.col(name_col)))
        .withColumn("_stripped",
            F.trim(F.regexp_replace(
                F.col("_upper"), cfg["strip_prefix_pattern"], ""
            ))
        )
        .withColumn("_denoised",
            F.trim(F.regexp_replace(
                F.col("_stripped"), cfg["strip_suffix_pattern"], ""
            ))
        )
        .withColumn("name_clean",
            F.initcap(F.trim(F.col("_denoised")))
        )
        .withColumn("name_keyword",
            F.initcap(F.trim(
                F.regexp_extract(F.col("_denoised"), r"^(\\\\S+)", 1)
            ))
        )
        .drop("_upper", "_stripped", "_denoised")
        .withColumnRenamed(id_col, "sor_id")
    )

result_schema = StructType([
    StructField("name_clean",         StringType(), True),
    StructField("name_keyword",       StringType(), True),
    StructField("candidates_json",    StringType(), True),
    StructField("raw_vector_score",   FloatType(),  True),
    StructField("top_candidate_name", StringType(), True),
])

def is_valid_name(name, max_digit_fraction):
    if not name or len(name.strip()) < 2:
        return False
    digit_ratio = sum(c.isdigit() for c in name) / len(name)
    return digit_ratio < max_digit_fraction

def make_vector_worker(cfg):
    endpoint       = cfg["vector_endpoint"]
    index_name     = cfg["vector_index"]
    name_col       = cfg["vector_name_col"]
    top_k          = cfg["vector_top_k"]
    boost_exact    = cfg["score_boost_exact"]
    boost_partial  = cfg["score_boost_partial"]
    boost_reverse  = cfg["score_boost_reverse"]
    max_digit_frac = cfg["max_digit_fraction"]

    def worker(iterator):
        import json
        vsc   = VectorSearchClient()
        index = vsc.get_index(endpoint_name=endpoint, index_name=index_name)
        for pdf in iterator:
            results = []
            for _, row in pdf.iterrows():
                name_clean   = str(row["name_clean"])   if row["name_clean"]   else ""
                name_keyword = str(row["name_keyword"]) if row["name_keyword"] else ""
                try:
                    r1 = index.similarity_search(
                        query_text=name_clean,
                        columns=[name_col],
                        num_results=top_k
                    ).get("result", {}).get("data_array", [])
                    r2 = []
                    if name_keyword and name_keyword.lower() != name_clean.lower():
                        r2 = index.similarity_search(
                            query_text=name_keyword,
                            columns=[name_col],
                            num_results=top_k
                        ).get("result", {}).get("data_array", [])
                    kw_upper   = name_keyword.upper()
                    full_upper = name_clean.upper()
                    seen = {}
                    for data, weight in [(r1, 1.0), (r2, 0.95)]:
                        for r in data:
                            c_name  = str(r[0]).strip()
                            v_score = float(r[-1])
                            if not is_valid_name(c_name, max_digit_frac):
                                continue
                            c_upper = c_name.upper()
                            if kw_upper and kw_upper in c_upper:
                                boost = boost_exact
                            elif full_upper and full_upper in c_upper:
                                boost = boost_partial
                            elif kw_upper and c_upper in kw_upper:
                                boost = boost_reverse
                            else:
                                boost = 1.0
                            final_score = round(v_score * 100 * boost * weight, 2)
                            if c_name not in seen or final_score > seen[c_name]:
                                seen[c_name] = final_score
                    processed = sorted(
                        [{"name": n, "score": s} for n, s in seen.items()],
                        key=lambda x: x["score"], reverse=True
                    )[:5]
                    top = processed[0] if processed else None
                    results.append({
                        "name_clean":         name_clean,
                        "name_keyword":       name_keyword,
                        "candidates_json":    json.dumps(processed),
                        "raw_vector_score":   top["score"] if top else 0.0,
                        "top_candidate_name": top["name"]  if top else "",
                    })
                except Exception as ex:
                    results.append({
                        "name_clean":         name_clean,
                        "name_keyword":       name_keyword,
                        "candidates_json":    "[]",
                        "raw_vector_score":   0.0,
                        "top_candidate_name": "",
                    })
            yield pd.DataFrame(results)
    return worker

def build_prompt_column():
    return F.concat_ws("",
        F.lit(
            "You are an entity resolution assistant.\\\\n"
            "Your job: given a raw company name string from a financial transaction, "
            "find the best matching legal entity from the provided candidates.\\\\n\\\\n"
        ),
        F.lit("RAW INPUT: "),    F.col("name_clean"),
        F.lit("\\\\nCANDIDATES: "), F.col("candidates_json"),
        F.lit(
            "\\\\n\\\\nINSTRUCTIONS:\\\\n"
            "1. The raw input may contain leading codes, product names, or abbreviations "
            "that are NOT the company name.\\\\n"
            "2. Pick the candidate that best represents the actual legal entity.\\\\n"
            "3. Higher score = more likely correct, but name similarity is primary signal.\\\\n"
            "4. If none match, output UNMATCHED.\\\\n"
            "5. Never output a numeric value or score.\\\\n\\\\n"
            "RESPOND IN THIS FORMAT ONLY:\\\\n"
            "[MATCH]: <legal entity name or UNMATCHED> | [REASON]: <one sentence>"
        )
    )

def run_pipeline(cfg, spark, log_fn=None):
    if log_fn is None:
        log_fn = print
    spark.catalog.clearCache()
    written_tables = []

    log_fn("Loading source tables...")
    source_dfs = []
    for src in cfg["sources"]:
        df = normalize(
            spark.table(src["table"]),
            src["name_col"],
            src["id_col"],
            cfg
        )
        df = df.withColumn("_output_table", F.lit(src["output_table"]))
        source_dfs.append(df)
    log_fn(f"Loaded {len(source_dfs)} source table(s)")

    if len(source_dfs) == 1:
        search_registry = source_dfs[0].select("name_clean", "name_keyword")
    else:
        search_registry = source_dfs[0].select("name_clean", "name_keyword")
        for d in source_dfs[1:]:
            search_registry = search_registry.union(
                d.select("name_clean", "name_keyword")
            )

    search_registry = search_registry.distinct().filter(
        F.col("name_clean").isNotNull() &
        (F.length(F.trim(F.col("name_clean"))) >= cfg["min_name_length"])
    )
    log_fn(f"{search_registry.count()} unique names to resolve")

    vector_worker    = make_vector_worker(cfg)
    registry_matches = (
        search_registry
        .repartition(cfg["vector_partitions"])
        .mapInPandas(vector_worker, schema=result_schema)
        .cache()
    )
    log_fn(f"Vector search complete — {registry_matches.count()} resolved")

    llm_model       = cfg["llm_model"]
    llm_max_tokens  = cfg["llm_max_tokens"]
    llm_temperature = cfg["llm_temperature"]

    answer_key = (
        registry_matches
        .withColumn("prompt", build_prompt_column())
        .repartition(cfg["llm_partitions"])
        .withColumn("llm_raw", F.expr(
            f"ai_query('{llm_model}', prompt, "
            f"map('max_tokens', '{llm_max_tokens}', "
            f"    'temperature', '{llm_temperature}')"
            f")"
        ))
        .cache()
    )
    answer_key.count()
    log_fn("LLM reasoning complete")

    for src, df_clean in zip(cfg["sources"], source_dfs):
        df_final = (
            df_clean
            .join(registry_matches, on="name_clean", how="left")
            .join(
                answer_key.select("name_clean", "llm_raw"),
                on="name_clean", how="left"
            )
            .withColumn("matched_name_extracted",
                F.trim(F.regexp_replace(
                    F.regexp_extract("llm_raw", r"\\\\\\\\[MATCH\\\\\\\\]:\\\\s*([^|]+)", 1),
                    r"[\\\\*\\\\[\\\\]\\\\\\"\\\\']", ""
                ))
            )
            .withColumn("match_explanation",
                F.trim(F.regexp_extract(
                    "llm_raw", r"\\\\\\\\[REASON\\\\\\\\]:\\\\s*(.*)", 1
                ))
            )
            .withColumn("childcompanyname",
                F.when(
                    F.col("matched_name_extracted").isNotNull()
                    & (F.length(F.trim(F.col("matched_name_extracted"))) >= cfg["min_name_length"])
                    & (F.upper(F.trim(F.col("matched_name_extracted"))) != "UNMATCHED"),
                    F.col("matched_name_extracted")
                ).when(
                    (F.col("raw_vector_score") >= cfg["score_fallback_min"])
                    & F.col("top_candidate_name").isNotNull()
                    & (F.length(F.trim(F.col("top_candidate_name"))) >= cfg["min_name_length"])
                    & (~F.col("top_candidate_name").rlike(r"^[\\\\d\\\\.\\\\-]+$")),
                    F.col("top_candidate_name")
                ).otherwise(F.lit("No Clear Match"))
            )
        )
        out_table = f"{cfg['output_database']}.{src['output_table']}"
        (
            df_final
            .select(["sor_id", "src_company_name_raw", "childcompanyname", "match_explanation"])
            .write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(out_table)
        )
        written_tables.append(out_table)
        log_fn(f"Written to {out_table}")

    log_fn(f"Done. {len(written_tables)} table(s) written.")
    return written_tables

from pyspark.sql import SparkSession
spark_sess = SparkSession.builder.getOrCreate()
results = run_pipeline(cfg, spark_sess, log_fn=print)
print("DONE:", results)
"""

# ── 4. STREAMLIT UI LAYOUT STRUCTURE ────────────────────────────────────
st.set_page_config(page_title="Entity Resolution Portal", layout="wide")
st.title("🏦 ACH Enterprise Entity Resolution Portal")
st.markdown("Clean and map multi-tenant bank transaction logs to canonical S&P Global entities.")

# ── 5. METADATA HELPERS VIA DATABRICKS SDK ──────────────────────────────
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

# ── 6. UI COMPONENT LAYOUT GRIDS ────────────────────────────────────────
with st.expander("🔍 Vector Search Configurations", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        vector_endpoint = st.text_input("Endpoint Link", value="talk2db_poc_endpoint")
        vector_index    = st.text_input("Index Catalog Location", value="dev.er_sp_metadata.spg_entity_master_index")
    with col2:
        vector_name_col = st.text_input("Index Field Match", value="childcompanyname")
        vector_top_k    = st.number_input("Top-K Candidates Pool", value=15, min_value=1)

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
        sources_config.append({"table": full_table, "name_col": name_col, "id_col": id_col, "output_table": out_name})
    st.divider()

st.button("➕ Link Another Source Data Grid", on_click=add_source)

with st.expander("⚙️ Fine-Tuning Algorithmic Bounds", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        score_boost_exact   = st.number_input("Keyword Boost", value=1.4, step=0.05)
        score_boost_partial = st.number_input("Partial String Boost", value=1.3, step=0.05)
        score_boost_reverse = st.number_input("Reverse Containment Boost", value=1.2, step=0.05)
    with col2:
        score_fallback_min  = st.number_input("Minimum Fallback Match Bound", value=90.0, step=1.0)
        min_name_length     = st.number_input("Floor Length Variable", value=2, step=1)
        max_digit_fraction  = st.number_input("Numeric Noise Filter Ratio", value=0.7, step=0.05)
    with col3:
        vector_partitions   = st.number_input("Vector Search Distributed Partitions", value=60, step=10)
        llm_partitions      = st.number_input("LLM Reason Worker Partitions", value=120, step=10)
        llm_max_tokens      = st.number_input("Max Sequence LLM Tokens Allocation", value=150, step=10)

# ── 7. COMPILING CONSTRAINTS AND DISPATCHING ────────────────────────────
st.subheader("🚀 Operational Execution Grid")

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

run_disabled = len(sources_config) == 0

if st.button("▶️ Initiate Entity Matching Job Loop", disabled=run_disabled, type="primary"):
    log_box = st.empty()
    logs = ["Initialising serverless pipeline execution context..."]
    log_box.code(logs[0])

    with st.spinner("Compiling UI configurations and dispatching cluster task thread..."):
        try:
            # Format our massive template code text block by injecting current UI parameters
            executable_script = PIPELINE_CODE.format(cfg_json=json.dumps(config_preview))
            
            logs.append("Submitting dynamically compiled payload to target task runner...")
            log_box.code("\n".join(logs))
            
            # API Handoff using SparkSubmitTask flags to force zero-notebook background execution
            run_poller = w.jobs.submit(
                run_name="entity_resolution_ui_invocation",
                tasks=[
                    Task(
                        task_key="er_pipeline_exec",
                        spark_submit_task=SparkSubmitTask(parameters=["-e", executable_script]),
                        existing_cluster_id=CLUSTER_ID
                    )
                ]
            ).result()
            
            st.success("🎉 Resolution loop complete! Clean targets compiled successfully inside target catalogs.")
            st.balloons()
            
        except Exception as e:
            st.error(f"Execution thread halted: {str(e)}")
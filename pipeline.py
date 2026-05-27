content = '''
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import *
from databricks.vector_search.client import VectorSearchClient

DEFAULT_CONFIG = {
    "vector_endpoint":      "talk2db_poc_endpoint",
    "vector_index":         "dev.er_sp_metadata.spg_entity_master_index",
    "vector_name_col":      "childcompanyname",
    "vector_top_k":         15,
    "llm_model":            "databricks-llama-4-maverick",
    "llm_max_tokens":       150,
    "llm_temperature":      0,
    "score_boost_exact":    1.4,
    "score_boost_partial":  1.3,
    "score_boost_reverse":  1.2,
    "score_fallback_min":   90.0,
    "vector_partitions":    60,
    "llm_partitions":       120,
    "output_database":      "dev.er_data",
    "strip_prefix_pattern": r"^(#[A-Z0-9]+\s+|[0-9]+[A-Z0-9]*\s+)",
    "strip_suffix_pattern": r"(?i)\\b(LLC|INC|CORP|LTD|SERVICES|S\\.P\\.A|B\\.V|K\\.K|DEFAULT|APP|APPS|PLATFORM|PAYMENT|PAYMENTS|PAY)\\b",
    "min_name_length":      2,
    "max_digit_fraction":   0.7,
    "sources":              [],
}

def normalize(df, name_col, id_col, cfg):
    return (
        df
        .withColumn("src_company_name_raw", F.col(name_col))
        .withColumn("_upper", F.upper(F.col(name_col)))
        .withColumn("_stripped",
            F.trim(F.regexp_replace(
                F.col("_upper"),
                cfg["strip_prefix_pattern"], ""
            ))
        )
        .withColumn("_denoised",
            F.trim(F.regexp_replace(
                F.col("_stripped"),
                cfg["strip_suffix_pattern"], ""
            ))
        )
        .withColumn("name_clean",
            F.initcap(F.trim(F.col("_denoised")))
        )
        .withColumn("name_keyword",
            F.initcap(F.trim(
                F.regexp_extract(F.col("_denoised"), r"^(\\S+)", 1)
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
                except Exception:
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
            "You are an entity resolution assistant.\\n"
            "Your job: given a raw company name string from a financial transaction, "
            "find the best matching legal entity from the provided candidates.\\n\\n"
        ),
        F.lit("RAW INPUT: "),    F.col("name_clean"),
        F.lit("\\nCANDIDATES: "), F.col("candidates_json"),
        F.lit(
            "\\n\\nINSTRUCTIONS:\\n"
            "1. The raw input may contain leading codes, product names, or abbreviations "
            "that are NOT the company name — use judgment to identify the core brand.\\n"
            "2. From the candidates list, pick the one that best represents the "
            "actual legal entity behind the raw input.\\n"
            "3. A candidate with a higher score is more likely to be correct, "
            "but use the name similarity as your primary signal.\\n"
            "4. If none of the candidates plausibly match the core brand in the input, "
            "output UNMATCHED.\\n"
            "5. Never output a numeric value or score as the match.\\n\\n"
            "RESPOND IN THIS FORMAT ONLY (no extra text):\\n"
            "[MATCH]: <legal entity name or UNMATCHED> | [REASON]: <one sentence>"
        )
    )

def run_pipeline(cfg, spark, log_fn=None):
    if log_fn is None:
        log_fn = print

    spark.catalog.clearCache()
    written_tables = []

    log_fn("Loading and normalising source tables...")
    source_dfs = []
    for src in cfg["sources"]:
        df = normalize(
            spark.table(src["table"]),
            name_col=src["name_col"],
            id_col=src["id_col"],
            cfg=cfg,
        )
        df = df.withColumn("_output_table", F.lit(src["output_table"]))
        source_dfs.append(df)
    log_fn(f"Loaded {len(source_dfs)} source table(s)")

    log_fn("Building unique name registry...")
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

    unique_count = search_registry.count()
    log_fn(f"{unique_count} unique names to resolve")

    log_fn("Running vector search...")
    vector_worker    = make_vector_worker(cfg)
    registry_matches = (
        search_registry
        .repartition(cfg["vector_partitions"])
        .mapInPandas(vector_worker, schema=result_schema)
        .cache()
    )
    resolved_count = registry_matches.count()
    log_fn(f"Vector search complete — {resolved_count} entities resolved")

    log_fn("Running LLM reasoning via ai_query...")
    answer_key = (
        registry_matches
        .withColumn("prompt", build_prompt_column())
        .repartition(cfg["llm_partitions"])
        .withColumn(
            "llm_raw",
            F.expr(
                f"ai_query("
                f"  \'{cfg[\'llm_model\']}\','
                f"  prompt,"
                f"  map(\'max_tokens\', \'{cfg[\'llm_max_tokens\']}\', "
                f"      \'temperature\', \'{cfg[\'llm_temperature\']}\')"
                f")"
            )
        )
        .cache()
    )
    answer_key.count()
    log_fn("LLM reasoning complete")

    for src, df_clean in zip(cfg["sources"], source_dfs):
        log_fn(f"Writing results for {src[\'table\']}...")
        df_final = (
            df_clean
            .join(registry_matches, on="name_clean", how="left")
            .join(answer_key.select("name_clean", "llm_raw"), on="name_clean", how="left")
        )
        df_final = (
            df_final
            .withColumn("matched_name_extracted",
                F.trim(F.regexp_replace(
                    F.regexp_extract("llm_raw", r"\\[MATCH\\]:\\s*([^|]+)", 1),
                    r"[\\*\\[\\]\\"\\']", ""
                ))
            )
            .withColumn("match_explanation",
                F.trim(F.regexp_extract("llm_raw", r"\\[REASON\\]:\\s*(.*)", 1))
            )
        )
        df_final = df_final.withColumn(
            "childcompanyname",
            F.when(
                F.col("matched_name_extracted").isNotNull()
                & (F.length(F.trim(F.col("matched_name_extracted"))) >= cfg["min_name_length"])
                & (F.upper(F.trim(F.col("matched_name_extracted"))) != "UNMATCHED"),
                F.col("matched_name_extracted")
            )
            .when(
                (F.col("raw_vector_score") >= cfg["score_fallback_min"])
                & F.col("top_candidate_name").isNotNull()
                & (F.length(F.trim(F.col("top_candidate_name"))) >= cfg["min_name_length"])
                & (~F.col("top_candidate_name").rlike(r"^[\\d\\.\\-]+$")),
                F.col("top_candidate_name")
            )
            .otherwise(F.lit("No Clear Match"))
        )

        out_table  = f"{cfg[\'output_database\']}.{src[\'output_table\']}"
        final_cols = ["sor_id", "src_company_name_raw", "childcompanyname", "match_explanation"]

        (
            df_final.select(final_cols)
            .write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(out_table)
        )
        written_tables.append(out_table)
        log_fn(f"Written to {out_table}")

    log_fn(f"Pipeline complete. {len(written_tables)} table(s) written.")
    return written_tables
'''

with open("/Workspace/Shared/ER-APP/pipeline.py", "w") as f:
    f.write(content)

print("pipeline.py written successfully!")

# Verify it works
import sys
sys.path.insert(0, "/Workspace/Shared/ER-APP")
import importlib
import pipeline
importlib.reload(pipeline)
print(dir(pipeline))
print("run_pipeline found:", hasattr(pipeline, "run_pipeline"))

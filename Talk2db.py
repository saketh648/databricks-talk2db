# Databricks notebook source
# Serverless requires these specific versions for the best stability
%pip install databricks-langchain databricks-vectorsearch langchain-community
%pip install databricks-langchain==0.17.0 langgraph
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC Initialization

# COMMAND ----------

from databricks_langchain import DatabricksVectorSearch

# 1. Configuration (Keep these as strings)
INDEX_NAME = "talk2db_poc.ai_metadata.table_description_index"
ENDPOINT_NAME = "talk2db_poc_endpoint"

# 2. Simplified Initialization
# Since it's a Managed Index, we don't need to specify 'text_column'
vectorstore = DatabricksVectorSearch(
    index_name=INDEX_NAME,
    endpoint=ENDPOINT_NAME,
    columns=["table_name", "business_rules"]
)

print("✅ 'vectorstore' is now linked and synchronized with Unity Catalog!")

# COMMAND ----------

# MAGIC %md
# MAGIC Agent_Logic

# COMMAND ----------

from databricks_langchain import ChatDatabricks
from langgraph.prebuilt import create_react_agent

# 1. The LLM (Using the 2026 Maverick model)
llm = ChatDatabricks(endpoint="databricks-llama-4-maverick", temperature=0)
system_instructions = """You are Talk2DB at U.S. Bank. 
1. CRITICAL: Only use 3-part table names (e.g., talk2db_poc.data_layer.customer_metrics). 
2. NEVER guess table names like 'customers'. 
3. If run_sql fails with a 'TABLE_OR_VIEW_NOT_FOUND' error, call 'schema_search' again to find the correct 3-part name.
4. Always execute the SQL immediately and show the data—never just show the query code."""

# 2. Tool: Schema Search
def schema_search(query: str):
    """Search for table names and business rules (like churn)."""
    return vectorstore.similarity_search(query, k=2)

# 3. Tool: SQL Execution
def run_sql(query: str):
    """Execute SQL on Spark with stability checks for Serverless."""
    try:
        # Keep-alive: Ensure the session is actually connected
        spark.sql("SELECT 1").collect() 
        
        # Strip any accidental backticks or markdown the AI might add
        clean_query = query.replace("```sql", "").replace("```", "").strip()
        
        df = spark.sql(clean_query).limit(10)
        result_df = df.toPandas()
        
        if result_df.empty:
            return "The query returned no results."
            
        return result_df.to_string(index=False)
    except Exception as e:
        return f"SQL Execution Error: {str(e)}"

# 4. The Agent
tools = [schema_search, run_sql]
# Re-create the agent with 'prompt' instead of 'state_modifier'
talk2db_agent = create_react_agent(
    llm, 
    tools, 
    prompt=system_instructions  # Changed from state_modifier to prompt
)

print("✅ Agent brain re-wired. Ready for the chat loop!")

# COMMAND ----------

# MAGIC %md
# MAGIC Chat_Interface

# COMMAND ----------

print("--- 🤖 Talk2DB Conversational Session ---")
chat_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]: break

    # Add context
    inputs = {"messages": chat_history + [("user", user_input)]}
    
    print("Thinking...")
    
    # We use 'invoke' but we need to make sure the agent 
    # knows it MUST return a final answer, not just a tool call.
    result = talk2db_agent.invoke(inputs)
    
    # The agent might take multiple steps. 
    # We want the VERY LAST message (the one where it has the data).
    final_msg = result["messages"][-1].content
    
    if not final_msg and hasattr(result["messages"][-1], 'tool_calls'):
        final_msg = "I've formulated the query, but I'm waiting for the data. Let me try executing it..."
    
    print(f"\nTalk2DB: {final_msg}")
    
    # Store the turn
    chat_history.append(("user", user_input))
    chat_history.append(("assistant", final_msg))
    
    print("\n" + "-"*40)

# COMMAND ----------

# MAGIC %md
# MAGIC
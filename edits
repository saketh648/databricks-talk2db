# Notebook Name: trigger_insert_notebook

# 1. Define the parameters the notebook expects from the app
dbutils.widgets.text("user_name", "")
dbutils.widgets.text("department", "")
dbutils.widgets.text("notes", "")

# 2. Retrieve the passed parameters
name = dbutils.widgets.get("user_name")
dept = dbutils.widgets.get("department")
notes = dbutils.widgets.get("notes")

# Basic validation guardrail
if not name:
    raise ValueError("User name parameter cannot be empty.")

# 3. Create the table if it doesn't exist
CATALOG = "main"
SCHEMA = "default"
TABLE_NAME = "user_inputs"
FULL_TABLE_PATH = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {FULL_TABLE_PATH} (
    user_name STRING,
    department STRING,
    submission_notes STRING,
    submission_timestamp TIMESTAMP
) USING DELTA;
""")

# 4. Insert the parameterized data
insert_query = f"""
    INSERT INTO {FULL_TABLE_PATH} 
    VALUES (:name, :dept, :notes, current_timestamp())
"""

spark.sql(insert_query, args={"name": name, "dept": dept, "notes": notes})
print(f"Successfully inserted record for {name}")

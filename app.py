import json
import os

with open("Template.json", "r") as f:
    template = json.load(f)

resources = template.get("resources", [])

for res in resources:
    res_type = res.get("type", "")
    props = res.get("properties", {})
    
    # Extract Notebooks
    if "notebooks" in res_type:
        name = res.get("name", "").split("/")[-1]
        folder_path = props.get("folder", {}).get("name", "")
        target_dir = os.path.join("databricks_export", folder_path)
        os.makedirs(target_dir, exist_ok=True)
        
        # Save as Jupyter Notebook or source Python
        file_path = os.path.join(target_dir, f"{name}.ipynb")
        notebook_content = {
            "cells": props.get("cells", []),
            "metadata": props.get("metadata", {}),
            "nbformat": 4,
            "nbformat_minor": 2
        }
        with open(file_path, "w", encoding="utf-8") as out:
            json.dump(notebook_content, out, indent=2)

    # Extract SQL Scripts
    elif "sqlscripts" in res_type:
        name = res.get("name", "").split("/")[-1]
        folder_path = props.get("folder", {}).get("name", "")
        target_dir = os.path.join("databricks_export", folder_path)
        os.makedirs(target_dir, exist_ok=True)
        
        sql_query = props.get("content", {}).get("query", "")
        file_path = os.path.join(target_dir, f"{name}.sql")
        with open(file_path, "w", encoding="utf-8") as out:
            out.write(sql_query)

print("Export completed! Files organized in ./databricks_export")

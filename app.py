import streamlit as st
import requests
import os

# 1. Grab the automated environment credentials provided by Databricks Apps
DBX_HOST = os.environ.get("DATABRICKS_HOST")
DBX_TOKEN = os.environ.get("DATABRICKS_TOKEN")

# Replace this with the Job ID you created in Step 2
JOB_ID = 123456789012345 

st.set_page_config(page_title="Workflow Trigger App", layout="centered")
st.title("📥 Submit Data via Workflow")

with st.form("input_form", clear_on_submit=True):
    user_name = st.text_input("Full Name")
    department = st.selectbox("Department", ["Engineering", "Data Science", "Product", "Operations"])
    notes = st.text_area("Notes/Comments")
    
    submit_button = st.form_submit_button(label="Trigger Write Job")

if submit_button:
    if not user_name.strip():
        st.warning("Please enter a name before submitting.")
    else:
        # API payload containing the parameters the notebook expects
        payload = {
            "job_id": JOB_ID,
            "notebook_params": {
                "user_name": user_name,
                "department": department,
                "notes": notes
            }
        }
        
        headers = {
            "Authorization": f"Bearer {DBX_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Trigger the run-now endpoint asynchronously
        api_url = f"https://{DBX_HOST}/api/2.1/jobs/run-now"
        
        with st.spinner("Communicating with Databricks Workflows..."):
            try:
                response = requests.post(api_url, json=payload, headers=headers)
                
                if response.status_code == 200:
                    run_id = response.json().get("run_id")
                    st.success(f"🚀 Job triggered successfully! Run ID: {run_id}")
                    st.toast("Data processing started in the background.", icon="⚙️")
                else:
                    st.error(f"Failed to trigger workflow: {response.text}")
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")

import streamlit as st
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

JOB_ID = 123456789012345

st.set_page_config(page_title="Workflow Trigger App", layout="centered")
st.title("📥 Submit Data via Workflow")

# Auth check in sidebar
try:
    me = w.current_user.me()
    st.sidebar.success(f"✅ Auth OK: {me.user_name}")
except Exception as e:
    st.sidebar.error(f"❌ Auth failed: {e}")

with st.form("input_form", clear_on_submit=True):
    user_name = st.text_input("Full Name")
    department = st.selectbox("Department", ["Engineering", "Data Science", "Product", "Operations"])
    notes = st.text_area("Notes/Comments")
    submit_button = st.form_submit_button(label="Trigger Write Job")

if submit_button:
    if not user_name.strip():
        st.warning("Please enter a name before submitting.")
    else:
        notebook_params = {
            "user_name": user_name,
            "department": department,
            "notes": notes,
        }

        with st.spinner("Triggering workflow via Databricks SDK..."):
            try:
                run_response = w.jobs.run_now(
                    job_id=JOB_ID,
                    notebook_params=notebook_params
                )

                run_id = run_response.response.run_id
                st.success(f"🚀 Job triggered successfully! Run ID: {run_id}")

            except Exception as e:
                st.error(f"Failed to trigger workflow: {e}")

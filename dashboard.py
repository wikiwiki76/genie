# dashboard.py
import json
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="UOB One – Customer Interest Advisor", layout="centered")

st.title("UOB One – Customer Interest Advisor")
st.caption("Select a Customer ID to view inputs and AI-generated recommendation.")

# ----------------------------
# Load data
# ----------------------------
customers_csv = Path("customers.csv")
results_csv   = Path("outputs/uob_one_interest_simulation_rows.csv")

if not customers_csv.exists() or not results_csv.exists():
    st.error("Missing data. Run `generate_interest.py` first to create outputs.")
    st.stop()

df_customers = pd.read_csv(customers_csv, dtype={"customer_id": str, "snap_date": str})
df_results   = pd.read_csv(results_csv, dtype={"customer_id": str})

# Merge on customer_id (left join to show even if no result)
df = df_customers.merge(df_results, on="customer_id", how="left", suffixes=("_inp", "_ai"))

# ----------------------------
# UI – Dropdown
# ----------------------------
ids = df["customer_id"].dropna().unique().tolist()
if not ids:
    st.warning("No customers found in customers.csv.")
    st.stop()

selected_id = st.selectbox("Customer ID", ids, index=0)
rec = df[df["customer_id"] == selected_id].iloc[0]

# ----------------------------
# Show Inputs
# ----------------------------
st.subheader("Customer Inputs")
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"**Name:** {rec['customer_name_inp']}")
    st.markdown(f"**ID:** {rec['customer_id']}")
    st.markdown(f"**Snap Date:** {rec['snap_date_inp']}")
with c2:
    try:
        st.markdown(f"**Avg Balance:** ${float(rec['avg_balance']):,.2f}")
    except Exception:
        st.markdown("**Avg Balance:** —")
    try:
        st.markdown(f"**Card Spend:** ${float(rec['card_spend']):,.2f}")
    except Exception:
        st.markdown("**Card Spend:** —")
    try:
        st.markdown(f"**Salary Credit:** ${float(rec['salary_credit']):,.2f}")
    except Exception:
        st.markdown("**Salary Credit:** —")
    try:
        st.markdown(f"**GIRO Count:** {int(rec['giro_count'])}")
    except Exception:
        st.markdown("**GIRO Count:** —")

st.divider()

# ----------------------------
# AI Output (flat highlights)
# ----------------------------
st.subheader("AI Summary")
bm = rec.get("banker_message", "")
if pd.isna(bm) or not str(bm).strip():
    st.warning("No AI summary found for this customer.")
else:
    st.text(bm)

# Current snapshot
st.markdown("**Current Status**")
c3, c4, c5 = st.columns(3)
c3.metric("Level", rec.get("current_level", ""))
c4.metric("Tier", rec.get("current_tier", ""))
try:
    curr_total = float(rec.get("current_total_interest_month", 0))
    c5.metric("Est. Interest (Month)", f"${curr_total:,.2f}")
except Exception:
    c5.metric("Est. Interest (Month)", "—")

# ----------------------------
# Full JSON + download
# ----------------------------
st.divider()
st.subheader("Full LLM JSON")
llm_json_str = rec.get("llm_json", "")
if pd.isna(llm_json_str) or not str(llm_json_str).strip():
    st.info("No JSON available.")
else:
    try:
        parsed = json.loads(llm_json_str)
        st.json(parsed)
        st.download_button(
            label="Download LLM JSON",
            data=json.dumps(parsed, indent=2, ensure_ascii=False),
            file_name=f"{rec['customer_id']}_llm_result.json",
            mime="application/json"
        )
    except Exception:
        st.code(str(llm_json_str)[:5000], language="json")

st.divider()

# Allow quick CSV downloads
st.subheader("Data Files")
with open(results_csv, "rb") as f:
    st.download_button(
        "Download Results CSV",
        data=f,
        file_name="uob_one_interest_simulation_rows.csv",
        mime="text/csv"
    )
with open(customers_csv, "rb") as f:
    st.download_button(
        "Download Customers CSV",
        data=f,
        file_name="customers.csv",
        mime="text/csv"
    )

#st.caption("Tip: Edit customers.csv and re-run generate_interest.py to refresh results.")

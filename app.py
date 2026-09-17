"""
Customer Lifetime Value (CLV) Prediction App
==============================================
Loads the two-stage CLV model (churn classifier + spend regressor) and
lets a user either score a single hypothetical customer, upload a CSV
of customers to score in bulk, or browse existing scored customers.

Expected folder structure — everything lives in ONE flat folder, no subfolder:

    clv_app/
        app.py
        stage1_classifier.pkl
        stage2_regressor.pkl
        feature_cols.json
        clv_tier_thresholds.json
        tier_summary.csv
        customer_results.csv

Run with:  streamlit run app.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------
# Load artifacts (cached so they only load once per session, not per click)
# All files are expected to sit in the SAME folder as this script.
# --------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    clf = joblib.load("stage1_classifier.pkl")
    reg = joblib.load("stage2_regressor.pkl")

    with open("feature_cols.json") as f:
        feature_cols = json.load(f)

    with open("clv_tier_thresholds.json") as f:
        tier_thresholds = json.load(f)

    tier_summary = pd.read_csv("tier_summary.csv", index_col=0)
    customer_results = pd.read_csv("customer_results.csv")

    return clf, reg, feature_cols, tier_thresholds, tier_summary, customer_results


clf, reg, feature_cols, tier_thresholds, tier_summary, customer_results = load_artifacts()


# --------------------------------------------------------------------
# Core scoring logic — shared by both the single-customer and bulk views
# --------------------------------------------------------------------

def assign_tier(clv_value, thresholds):
    """Bucket a CLV value using the SAME quartile cutoffs from training."""
    q25, q50, q75 = thresholds["0.25"], thresholds["0.5"], thresholds["0.75"]
    if clv_value <= q25:
        return "Low"
    elif clv_value <= q50:
        return "Medium"
    elif clv_value <= q75:
        return "High"
    else:
        return "Top"


def score_customers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a dataframe containing the raw RFM feature columns and returns
    it with purchase_probability, predicted_spend_if_buying, expected_clv,
    and clv_tier attached.
    """
    X = df[feature_cols]

    purchase_proba = clf.predict_proba(X)[:, 1]
    predicted_spend_log = reg.predict(X)
    predicted_spend_dollars = np.expm1(predicted_spend_log)
    expected_clv = purchase_proba * predicted_spend_dollars

    out = df.copy()
    out["purchase_probability"] = purchase_proba
    out["predicted_spend_if_buying"] = predicted_spend_dollars
    out["expected_clv"] = expected_clv
    out["clv_tier"] = out["expected_clv"].apply(lambda v: assign_tier(v, tier_thresholds))
    return out


# --------------------------------------------------------------------
# App layout
# --------------------------------------------------------------------

st.set_page_config(page_title="CLV Prediction", layout="wide")
st.title("Customer Lifetime Value Prediction")
st.caption(
    "Two-stage model: predicts probability of a future purchase, "
    "then predicts spend amount conditional on purchasing."
)

tab1, tab2, tab3 = st.tabs(["Score a Customer", "Bulk Upload", "Explore Segments"])

# ---------------- Tab 1: Single customer form ----------------
with tab1:
    st.subheader("Enter a customer's RFM profile")
    st.write("Fill in the fields below to get a live CLV prediction.")

    col1, col2 = st.columns(2)
    input_values = {}

    # Build a numeric input for every feature the model expects
    for i, feature in enumerate(feature_cols):
        target_col = col1 if i % 2 == 0 else col2
        input_values[feature] = target_col.number_input(
            feature.replace("_", " ").title(),
            min_value=0.0,
            value=0.0,
            step=1.0,
        )

    if st.button("Predict CLV", type="primary"):
        input_df = pd.DataFrame([input_values])
        scored = score_customers(input_df)
        row = scored.iloc[0]

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Purchase Probability", f"{row['purchase_probability']:.1%}")
        m2.metric("Predicted Spend (if buying)", f"${row['predicted_spend_if_buying']:,.2f}")
        m3.metric("Expected CLV", f"${row['expected_clv']:,.2f}")

        st.info(f"CLV Tier: **{row['clv_tier']}**")

# ---------------- Tab 2: Bulk CSV upload ----------------
with tab2:
    st.subheader("Score multiple customers at once")
    st.write(f"Upload a CSV containing these columns: `{', '.join(feature_cols)}`")

    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        missing_cols = [c for c in feature_cols if c not in input_df.columns]

        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            scored = score_customers(input_df)
            st.success(f"Scored {len(scored)} customers.")
            st.dataframe(
                scored.sort_values("expected_clv", ascending=False),
                use_container_width=True,
            )

            csv_out = scored.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download scored results",
                data=csv_out,
                file_name="scored_customers.csv",
                mime="text/csv",
            )

# ---------------- Tab 3: Explore existing segments ----------------
with tab3:
    st.subheader("CLV Tier Profiles (from training data)")
    st.write("Average RFM profile per CLV tier — what distinguishes high-value customers.")
    st.dataframe(tier_summary, use_container_width=True)

    st.divider()

    st.subheader("Existing Scored Customers")
    tier_filter = st.multiselect(
        "Filter by tier",
        options=customer_results["clv_tier"].unique(),
        default=list(customer_results["clv_tier"].unique()),
    )
    filtered = customer_results[customer_results["clv_tier"].isin(tier_filter)]
    st.dataframe(
        filtered.sort_values("expected_clv", ascending=False),
        use_container_width=True,
    )

    st.divider()
    st.subheader("Expected CLV Distribution")
    st.bar_chart(customer_results["expected_clv"].value_counts(bins=30).sort_index())

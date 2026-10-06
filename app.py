import pandas as pd
import streamlit as st
from Prediction_Model import data_handling
from datetime import datetime


# ============================================================
# LOAD MODEL + FEATURE ENGINEERING PIPELINE
# ============================================================

fe_pipe = data_handling.load_pipeline("fe_pipeline_fitted")
model = data_handling.load_pipeline("XBG_model")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SIMPLE CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */

    section[data-testid="stSidebar"] {
        background-color: #0f172a;
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    /* Sidebar divider */

    section[data-testid="stSidebar"] hr {
        border-color: #334155;
    }

    /* Main title */

    .main-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .main-subtitle {
        font-size: 15px;
        color: #64748b;
        margin-bottom: 20px;
    }

    /* Assessment heading */

    .assessment-title {
        font-size: 24px;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🏦 CREDIT RISK")

    st.caption("ANALYTICS PLATFORM")

    st.divider()

    st.subheader("AI-Powered Lending")

    st.write(
        "Evaluate borrower risk using machine learning "
        "and data-driven credit assessment."
    )

    st.divider()

    # Model information

    st.metric(
        label="MODEL",
        value="XGBoost"
    )

    st.caption("Credit Default Classifier")

    # Dataset information

    st.metric(
        label="DATASET",
        value="256K+"
    )

    st.caption("Borrower Records")

    # Accuracy information

    st.metric(
        label="TEST ACCURACY",
        value="89%"
    )

    st.caption("Model Performance")

    st.divider()

    st.subheader("Risk Classification")

    st.write("🟢 Low Risk")
    st.caption("Default probability below 30%")

    st.write("🟡 Medium Risk")
    st.caption("Default probability from 30% to 60%")

    st.write("🔴 High Risk")
    st.caption("Default probability above 60%")

    st.divider()

    st.caption("ML CREDIT ASSESSMENT")
    st.caption("Data → Features → Model → Risk")


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏦 Credit Default Risk Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'Machine-learning powered borrower risk assessment'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# HERO IMAGE
# ============================================================

st.image(
    "notebooks/Designer.jpeg",
    use_column_width=True
)


# ============================================================
# DESCRIPTION
# ============================================================

st.subheader("Assess Borrower Credit Risk")

st.write(
    "Enter the borrower and loan details below to "
    "estimate the probability of loan default."
)


# ============================================================
# APPLICATION FORM
# ============================================================

with st.form("loan_form"):

    st.subheader(
        "Please Fill in Your Loan Application Details"
    )

    col1, col2 = st.columns(2)


    # ========================================================
    # LEFT COLUMN
    # ========================================================

    with col1:

        loan_amnt = st.number_input(
            "Loan Amount",
            min_value=0.0,
            value=10000.0
        )

        term = st.selectbox(
            "Term",
            [
                " 36 months",
                " 60 months"
            ]
        )

        grade = st.selectbox(
            "Grade",
            [
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G"
            ]
        )

        sub_grade = st.selectbox(
            "Sub Grade",
            [
                f"{grade}{i}"
                for grade in "ABCDEFG"
                for i in range(1, 6)
            ]
        )

        emp_length = st.selectbox(
            "Employment Length",
            [
                "< 1 year",
                "1 year",
                "2 years",
                "3 years",
                "4 years",
                "5 years",
                "6 years",
                "7 years",
                "8 years",
                "9 years",
                "10+ years"
            ]
        )

        home_ownership = st.selectbox(
            "Home Ownership",
            [
                "RENT",
                "OWN",
                "MORTGAGE",
                "OTHER"
            ]
        )

        annual_inc = st.number_input(
            "Annual Income ($)",
            min_value=0.0,
            value=50000.0
        )


    # ========================================================
    # RIGHT COLUMN
    # ========================================================

    with col2:

        verification_status = st.selectbox(
            "Verification Status",
            [
                "Verified",
                "Not Verified",
                "Source Verified"
            ]
        )

        purpose = st.selectbox(
            "Purpose",
            [
                "debt_consolidation",
                "credit_card",
                "home_improvement",
                "other"
            ]
        )

        title = st.text_input(
            "Loan Title",
            value="Debt Consolidation"
        )

        dti = st.number_input(
            "Debt-to-Income Ratio",
            min_value=0.0,
            value=15.0
        )

        last_pymnt_amnt = st.number_input(
            "Last Paid EMI Amount",
            min_value=0.0,
            value=200.0
        )

        initial_list_status = st.selectbox(
            "Initial List Status",
            [
                "Whole Loan",
                "Fractional Loan"
            ]
        )

        addr_state = st.selectbox(
            "State",
            [
                "AL", "AK", "AZ", "AR", "CA", "CO",
                "CT", "DE", "FL", "GA", "HI", "ID",
                "IL", "IN", "IA", "KS", "KY", "LA",
                "ME", "MD", "MA", "MI", "MN", "MS",
                "MO", "MT", "NE", "NV", "NH", "NJ",
                "NM", "NY", "NC", "ND", "OH", "OK",
                "OR", "PA", "RI", "SC", "SD", "TN",
                "TX", "UT", "VT", "VA", "WA", "WV",
                "WI", "WY"
            ]
        )


    # ========================================================
    # SUBMIT
    # ========================================================

    submit_button = st.form_submit_button(
        "🔍 Predict Loan Status",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if submit_button:

    # --------------------------------------------------------
    # INPUT DATA
    # --------------------------------------------------------

    input_fields = {

        "loan_amnt": loan_amnt,

        "term": term,

        "grade": grade,

        "sub_grade": sub_grade,

        "emp_length": emp_length,

        "home_ownership": home_ownership,

        "annual_inc": annual_inc,

        "verification_status":
            verification_status,

        "purpose":
            purpose,

        "title":
            title,

        "dti":
            dti,

        "last_pymnt_amnt":
            last_pymnt_amnt,

        "initial_list_status":
            (
                "w"
                if initial_list_status == "Whole Loan"
                else "f"
            ),

        "addr_state":
            addr_state,

        "mths_since_last_delinq":
            None,

        "mths_since_last_major_derog":
            None,

        "mths_since_last_record":
            None,

        "issue_d":
            datetime.today().strftime("%b-%Y"),

        "emp_title":
            "Unknown",

        "earliest_cr_line":
            "06-1997",

        "open_acc":
            0.0,

        "pub_rec":
            0.0,

        "revol_bal":
            0.0,

        "revol_util":
            0.0,

        "total_acc":
            0.0,

        "zip_code":
            "00000",

        "delinq_2yrs":
            0.0,

        "inq_last_6mths":
            0.0,

        "collections_12_mths_ex_med":
            0.0,

        "open_acc_6m":
            0.0,

        "open_il_6m":
            0.0,

        "open_il_12m":
            0.0,

        "open_il_24m":
            0.0,

        "mths_since_rcnt_il":
            0.0,

        "total_bal_il":
            0.0,

        "il_util":
            0.0,

        "open_rv_12m":
            0.0,

        "open_rv_24m":
            0.0,

        "max_bal_bc":
            0.0,

        "all_util":
            0.0,

        "total_rev_hi_lim":
            0.0,

        "inq_fi":
            0.0,

        "total_cu_tl":
            0.0,

        "inq_last_12m":
            0.0,

        "tot_coll_amt":
            0.0,

        "tot_cur_bal":
            0.0,

        "application_type":
            "Individual",

        "lat":
            47.7511,

        "lng":
            120.7401
    }


    # --------------------------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------------------------

    data = pd.DataFrame(
        [input_fields]
    )


    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    transformed_data = fe_pipe.transform(
        data
    )


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(
        transformed_data
    )[0]

    probabilities = model.predict_proba(
        transformed_data
    )[0]


    # --------------------------------------------------------
    # DEFAULT PROBABILITY
    # --------------------------------------------------------

    default_probability = float(
        probabilities[1] * 100
    )


    # --------------------------------------------------------
    # RISK CLASSIFICATION
    # --------------------------------------------------------

    if default_probability < 30:

        risk_level = "Low Risk"

    elif default_probability < 60:

        risk_level = "Medium Risk"

    else:

        risk_level = "High Risk"


    # --------------------------------------------------------
    # LOAN DECISION
    # --------------------------------------------------------

    if prediction == 0:

        decision = "Approved"

    else:

        decision = "Rejected"


    # ========================================================
    # RESULTS
    # ========================================================

    st.divider()

    st.subheader(
        "📊 Credit Risk Assessment"
    )

    result1, result2, result3 = st.columns(3)


    with result1:

        st.metric(
            "Default Probability",
            f"{default_probability:.1f}%"
        )


    with result2:

        st.metric(
            "Risk Level",
            risk_level
        )


    with result3:

        st.metric(
            "Decision",
            decision
        )


    # ========================================================
    # RISK BAR
    # ========================================================

    st.write("### Default Risk")

    st.progress(
        min(default_probability / 100, 1.0)
    )


    # ========================================================
    # DECISION MESSAGE
    # ========================================================

    if prediction == 0:

        st.success(
            f"✓ Loan application approved. "
            f"Estimated default probability: "
            f"{default_probability:.1f}%."
        )

    else:

        st.error(
            f"✕ Loan application rejected. "
            f"Estimated default probability: "
            f"{default_probability:.1f}%."
        )


    # ========================================================
    # RISK EXPLANATION
    # ========================================================

    if risk_level == "Low Risk":

        st.info(
            "The applicant falls within the low-risk "
            "segment based on the model's estimated "
            "probability of default."
        )

    elif risk_level == "Medium Risk":

        st.warning(
            "The applicant falls within the medium-risk "
            "segment. Additional credit assessment may "
            "be appropriate."
        )

    else:

        st.warning(
            "The applicant falls within the high-risk "
            "segment based on the model's estimated "
            "probability of default."
        )
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import uvicorn
from Prediction_Model import data_handling
import datetime

app = FastAPI()

# Define a Pydantic model for the new features, with non-important ones set as optional
class LoanFeatures(BaseModel):
    loan_amnt: float = 5000
    term: str = " 36 months"
    grade: str = "A"
    sub_grade: str = "A1"
    emp_length: str = "10+ years"
    home_ownership: str = "OWN"
    annual_inc: float = 100000
    verification_status: str = "Verified"
    purpose: str = "credit_card"
    title: str = "Credit Card"
    dti: float = 8
    last_pymnt_amnt: float = 500
    initial_list_status: str = "w"
    addr_state: str = "CA"

    mths_since_last_delinq: float | None = None
    mths_since_last_major_derog: float | None = None
    mths_since_last_record: float | None = None

    issue_d: str = "Sep-2026"
    emp_title: str = "Engineer"
    earliest_cr_line: str = "06-1997"

    open_acc: float = 10
    pub_rec: float = 0
    revol_bal: float = 3000
    revol_util: float = 15
    total_acc: float = 25
    zip_code: str = "00000"

    delinq_2yrs: float = 0
    inq_last_6mths: float = 0
    collections_12_mths_ex_med: float = 0

    open_acc_6m: float = 0
    open_il_6m: float = 0
    open_il_12m: float = 0
    open_il_24m: float = 0

    mths_since_rcnt_il: float = 0
    total_bal_il: float = 0
    il_util: float = 0

    open_rv_12m: float = 0
    open_rv_24m: float = 0
    max_bal_bc: float = 0
    all_util: float = 0

    total_rev_hi_lim: float = 20000
    inq_fi: float = 0
    total_cu_tl: float = 0
    inq_last_12m: float = 0

    tot_coll_amt: float = 0
    tot_cur_bal: float = 20000

    application_type: str = "Individual"

    lat: float = 47.7511
    lng: float = 120.7401

    model_config = {
        "json_schema_extra": {
            "example": {
                "loan_amnt": 5000,
                "term": " 36 months",
                "grade": "A",
                "sub_grade": "A1",
                "emp_length": "10+ years",
                "home_ownership": "OWN",
                "annual_inc": 100000,
                "verification_status": "Verified",
                "purpose": "credit_card",
                "title": "Credit Card",
                "dti": 8,
                "last_pymnt_amnt": 500,
                "initial_list_status": "w",
                "addr_state": "CA",
                "mths_since_last_delinq": None,
                "mths_since_last_major_derog": None,
                "mths_since_last_record": None,
                "issue_d": "Sep-2026",
                "emp_title": "Engineer",
                "earliest_cr_line": "06-1997",
                "open_acc": 10,
                "pub_rec": 0,
                "revol_bal": 3000,
                "revol_util": 15,
                "total_acc": 25,
                "zip_code": "00000",
                "delinq_2yrs": 0,
                "inq_last_6mths": 0,
                "collections_12_mths_ex_med": 0,
                "open_acc_6m": 0,
                "open_il_6m": 0,
                "open_il_12m": 0,
                "open_il_24m": 0,
                "mths_since_rcnt_il": 0,
                "total_bal_il": 0,
                "il_util": 0,
                "open_rv_12m": 0,
                "open_rv_24m": 0,
                "max_bal_bc": 0,
                "all_util": 0,
                "total_rev_hi_lim": 20000,
                "inq_fi": 0,
                "total_cu_tl": 0,
                "inq_last_12m": 0,
                "tot_coll_amt": 0,
                "tot_cur_bal": 20000,
                "application_type": "Individual",
                "lat": 47.7511,
                "lng": 120.7401
            }
        }
    }

# Load feature engineering pipeline and model
fe_pipe = data_handling.load_pipeline("fe_pipeline_fitted")
model = data_handling.load_pipeline("XBG_model")

@app.get('/')
def index():
    return {'message': 'Welcome to Loan Prediction App'}

@app.post('/predict')
async def predict(loan_data: LoanFeatures):
    """
    Predicts whether a loan should be approved or not based on the input data.
    """
    # Extract the data from the input
    data = loan_data.model_dump()

    df = pd.DataFrame([data])
    # Apply the feature engineering pipeline and make predictions
    transformed_data = fe_pipe.transform(df)
    pred = model.predict(transformed_data)
    
    # Return prediction result
    if pred[0] == 0:
        return {'Status of Loan Application': 'Approved'}
    else:
        return {'Status of Loan Application': 'Rejected'}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)

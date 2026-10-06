# Credit Risk Prediction System

A machine learning based credit risk assessment system that predicts the likelihood of borrower default and classifies applicants into different risk categories.

The project combines a feature engineering pipeline, XGBoost classification model, Streamlit interface, and FastAPI REST API to provide an end-to-end credit risk prediction workflow.

## Project Overview

Credit risk assessment is an important component of lending and financial decision-making. The objective of this project is to build a machine learning pipeline that can process borrower and loan information, estimate default probability, and provide a risk-based lending decision.

The system supports:

- Borrower risk prediction
- Default probability estimation
- Low, medium, and high-risk classification
- Loan approval/rejection prediction
- Interactive Streamlit interface
- REST API using FastAPI
- Docker-based deployment

## Model

The project uses an **XGBoost classifier** for credit default prediction.

The modeling pipeline includes:

1. Data preprocessing
2. Feature engineering
3. Missing-value handling
4. Categorical feature encoding
5. Feature selection
6. Feature scaling
7. XGBoost model training
8. Classification threshold tuning
9. Probability-based risk classification

### Model Performance

The current model achieves approximately:

| Metric | Test Performance |
|---|---:|
| Accuracy | ~89% |
| Default Class F1 | ~0.74 |
| Default Class Recall | ~0.81 |

Performance can vary depending on the training configuration and feature-engineering parameters.

## Dataset

The project uses a reduced Lending Club loan dataset containing approximately **257,000 borrower records**.

The dataset contains borrower, loan, credit history, and repayment-related attributes.

The target variable is:

```text``` 
```loan_status```

The data is divided into training and testing sets for model development and evaluation.
System Architecture
                    Borrower / Loan Information
                              |
                              v
                    Feature Engineering
                              |
                              v
                  Preprocessing Pipeline
                              |
                              v
                       XGBoost Model
                              |
                 +------------+------------+
                 |                         |
                 v                         v
          Default Probability        Classification
                 |                         |
                 +------------+------------+
                              |
                              v
                     Risk Assessment
                              |
              +---------------+---------------+
              |                               |
              v                               v
       Streamlit Application          FastAPI REST API

Application
Streamlit
The Streamlit application provides an interactive interface where users can enter borrower and loan information.
The application returns:
- Default probability
- Risk level
- Loan decision
Risk categories used in the application:
Default Probability	Risk Level
< 30%	Low Risk
30–60%	Medium Risk
> 60%	High Risk


FastAPI
The project also exposes the prediction model through a REST API.
Main endpoints:
GET /
POST /predict

Interactive API documentation is automatically available through Swagger UI:
http://127.0.0.1:8000/docs

Project Structure
Credit-Risk-Project/
│
├── data/
│   └── loan_reduced.csv
│
├── notebooks/
│   ├── Designer.jpeg
│   └── model_prototyping.ipynb
│
├── Prediction_Model/
│   ├── config.py
│   ├── data_handling.py
│   ├── evaluation.py
│   ├── FE_pipeline.py
│   ├── get_features.py
│   ├── plotting.py
│   ├── predict.py
│   ├── train.py
│   └── trained_models/
│       ├── XBG_model.pkl
│       ├── fe_pipeline_fitted.pkl
│       └── target_pipeline_fitted.pkl
│
├── tests/
│
├── app.py
├── fastapi_app.py
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md

Technologies Used
Programming
- Python
- SQL concepts
- Pandas
- NumPy
Machine Learning
- XGBoost
- Scikit-learn
- Optuna
- Feature engineering
- Classification threshold tuning
Application
- Streamlit
- FastAPI
- Pydantic
- Uvicorn
Deployment
- Docker
Development
- Jupyter Notebook
- Git
- GitHub
Installation
Clone the repository:
git clone https://github.com/vKrish14/Credit-Risk-Project.git
cd Credit-Risk-Project

Create and activate a virtual environment:
Windows
python -m venv .venv
.venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

Running the Streamlit Application
Run:
python -m streamlit run app.py

The application will be available at:
http://localhost:8501

Running the FastAPI Backend
Start the API:
python -m uvicorn fastapi_app:app --reload

The API will be available at:
http://127.0.0.1:8000

Swagger documentation:
http://127.0.0.1:8000/docs

Running with Docker
Build the Docker image:
docker build -t credit-risk-prediction .

Run the container:
docker run -p 8000:8000 credit-risk-prediction

Model Pipeline
The trained artifacts used by the application are stored in:
Prediction_Model/trained_models/

The main artifacts are:
XBG_model.pkl
fe_pipeline_fitted.pkl
target_pipeline_fitted.pkl

The application loads the feature engineering pipeline before passing the transformed data to the trained XGBoost model.
Risk Decision Workflow
For a new borrower:
Input borrower information
          |
          v
Feature transformation
          |
          v
XGBoost prediction
          |
          v
Default probability
          |
          v
Risk classification
          |
          v
Loan decision

Limitations
This project is intended as a machine learning demonstration and should not be considered a production-ready credit underwriting system.
A production deployment would require additional work including:
- Strict feature availability checks at loan origination
- Data leakage validation
- Model calibration
- Fairness and bias assessment
- Model stability monitoring
- Feature drift monitoring
- Explainability and auditability
- Secure API authentication
- Data privacy controls
- Model versioning and governance
- Automated retraining and validation pipelines
Future Improvements
Potential extensions include:
- SHAP-based prediction explanations
- Real-time model monitoring
- Feature drift detection
- Automated model retraining
- Model registry integration
- Cloud deployment
- Authentication and authorization for the API
- Batch prediction pipelines
- Advanced risk segmentation
- Model calibration and probability adjustment
Disclaimer
This project is intended for educational, portfolio, and interview demonstration purposes. Predictions generated by the model should not be used as the sole basis for real-world lending decisions.

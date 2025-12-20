# Titanic Survival Prediction

Binary classification model to predict passenger survival on the Titanic disaster

Author: Shreyas Aravind  
Course: 503 Final Project  
Date: December 2024

## Project Overview

Machine learning system that predicts whether a Titanic passenger would survive based on demographic and ticket information.

Best Model: Random Forest Classifier  
F1-Score: 0.7385  
Accuracy: 81.01%

## Live Deployment

- Streamlit App: http://159.203.81.240:8501
- API Endpoint: http://159.203.81.240:8000
- API Documentation: http://159.203.81.240:8000/docs

## Dataset

Source: Titanic dataset (Kaggle)  
Samples: 891 passengers  
Target Variable: Survived (0 = Died, 1 = Survived)

Features Used:
- Age, Sex, Passenger Class
- Fare, Siblings/Spouses, Parents/Children
- Port of Embarkation

## Database

Normalized 3NF Database with 3 tables:

1. passengers - Core demographic information
2. tickets - Ticket class, fare, and family size
3. embarkation - Port of embarkation details

Implementation:
- SQLite database
- Foreign key constraints
- SQL JOIN queries for data retrieval

## Machine Learning Experiments

16 Experiments testing 4 algorithms with variations:

Algorithms Tested:
1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. LightGBM

Variations:
- With/Without PCA (dimensionality reduction)
- With/Without Hyperparameter Tuning (GridSearchCV)

Results Summary:

| Experiment | Algorithm | PCA | Tuning | F1-Score | Accuracy |
|------------|-----------|-----|--------|----------|----------|
| 5 | RandomForest | No | No | 0.7385 | 81.01% |
| 14 | LightGBM | No | Yes | 0.7313 | 79.89% |
| 13 | LightGBM | No | No | 0.7313 | 79.89% |

Best Model: Random Forest without PCA or tuning

All experiment results tracked in DagsHub: https://dagshub.com/ShreyasAravind/titanic-classification

## Deployment Architecture

Technology Stack:
- Backend: FastAPI
- Frontend: Streamlit
- Containerization: Docker & Docker Compose
- Cloud Platform: DigitalOcean
- Experiment Tracking: DagsHub + MLflow

## Local Development

Prerequisites:
- Python 3.12
- Docker & Docker Compose
- Git

Setup Instructions:
```bash
git clone https://github.com/ShreyasAravind/housing_app_fall25.git
cd housing_app_fall25
docker-compose up -d
```

Access applications:
- Streamlit: http://localhost:8501
- API: http://localhost:8000

Run Jupyter Notebook:
```bash
jupyter notebook
```

Open: Titanic_classsification.ipynb

## Project Structure
```
housing_app_fall25/
├── api/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── streamlit/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── data/
│   ├── train.csv
│   └── titanic.db
├── models/
│   ├── 01_LogisticRegression_PCA-False_Tuning-False.pkl
│   ├── 05_RandomForest_PCA-False_Tuning-False.pkl
│   └── (14 more models)
├── notebooks/
│   └── Titanic_classsification.ipynb
├── experiment_results.csv
├── docker-compose.yml
└── README.md
```

## API Usage

Health Check:
```bash
curl http://159.203.81.240:8000/health
```

Make Prediction:
```bash
curl -X POST http://159.203.81.240:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [{
      "pclass": 3,
      "sex": "male",
      "age": 22,
      "siblings_spouses": 1,
      "parents_children": 0,
      "fare": 7.25,
      "port_code": "S"
    }]
  }'
```

Response:
```json
{
  "predictions": ["Died"],
  "probabilities": [0.30],
  "count": 1
}
```

## Model Performance

Best Model Metrics:
- Algorithm: Random Forest (n_estimators=100)
- F1-Score: 0.7385
- Accuracy: 81.01%
- Cross-Validation F1: 0.7344 (±0.0131)

Key Findings:
- Gender was the strongest predictor
- Passenger class significantly affected survival
- PCA did not improve performance
- Simpler models performed best

## Links

- GitHub Repository: https://github.com/ShreyasAravind/housing_app_fall25
- DagsHub Experiments: https://dagshub.com/ShreyasAravind/titanic-classification
- Live Demo: http://159.203.81.240:8501

## Requirements Met

- Classification Dataset: Titanic survival (binary classification)
- Normalized Database: 3NF with 3 tables, SQL JOINs
- 16 ML Experiments: 4 algorithms × 4 variations
- DagsHub Integration: All experiments logged with F1-scores
- FastAPI Deployment: Model serving with /predict endpoint
- Streamlit UI: User-friendly prediction interface
- Docker Deployment: Containerized with docker-compose
- Cloud Hosting: Deployed on DigitalOcean

## Project Score

Total Points: 200

| Component | Points | Status |
|-----------|--------|--------|
| Database (3NF) | 25 | Complete |
| ML Experiments (16) | 80 | Complete |
| DagsHub Integration | 20 | Complete |
| FastAPI Deployment | 30 | Complete |
| Streamlit UI | 30 | Complete |
| Documentation | 15 | Complete |

## Contact

Shreyas Aravind  
GitHub: ShreyasAravind

Built with FastAPI, Streamlit, Docker, and deployed on DigitalOcean

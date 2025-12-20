# Titanic Survival Classification Project

**Student:** Shreyas Aravind  
**Course:** 503 Final Project  
**Date:** December 2024

## Project Description

This project implements a binary classification system to predict whether passengers would survive the Titanic disaster. The system includes a normalized database, machine learning experiments, and a deployed web application.

## Dataset

**Source:** Titanic dataset  
**Total Passengers:** 891  
**Target Variable:** Survived (0 = Died, 1 = Survived)

**Features:**
- Passenger Class (1st, 2nd, 3rd)
- Sex (male, female)
- Age
- Number of Siblings/Spouses aboard
- Number of Parents/Children aboard
- Fare paid
- Port of Embarkation (Southampton, Cherbourg, Queenstown)

## Database Implementation

Created a normalized 3NF database with three tables:

**Table 1: passengers**
- passenger_id (Primary Key)
- name
- age
- sex
- survived
- ticket_id (Foreign Key)
- embark_id (Foreign Key)

**Table 2: tickets**
- ticket_id (Primary Key)
- pclass
- fare
- ticket_number
- siblings_spouses
- parents_children

**Table 3: embarkation**
- embark_id (Primary Key)
- port_code
- port_name

**SQL JOIN Query:** Successfully retrieved all passenger data by joining the three tables.

## Machine Learning Experiments

Conducted 16 experiments testing 4 different algorithms:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. LightGBM

Each algorithm was tested with 4 variations:
- Without PCA, Without Tuning
- Without PCA, With Tuning
- With PCA, Without Tuning
- With PCA, With Tuning

**Results:** All 16 experiments completed with F1-scores calculated and saved.

**Best performing model:** Random Forest without PCA or tuning (F1-Score: 0.7385)

All experiment results are tracked in DagsHub and saved in `experiment_results.csv`.

## Deployment

The application is deployed on DigitalOcean using Docker containers.

**Live URLs:**
- Streamlit Application: http://159.203.81.240:8501
- API Endpoint: http://159.203.81.240:8000
- API Documentation: http://159.203.81.240:8000/docs

**Technology Stack:**
- FastAPI for backend API
- Streamlit for user interface
- Docker and Docker Compose for containerization
- DigitalOcean for cloud hosting

## How to Use the Application

Visit the Streamlit application at http://159.203.81.240:8501

Enter passenger information:
- Name
- Ticket Class (1, 2, or 3)
- Sex (male or female)
- Age
- Number of siblings/spouses
- Number of parents/children
- Fare amount
- Port of embarkation

Click "Predict Survival" to see whether the passenger would have survived.

## Project Files

- `Titanic_classsification.ipynb` - Jupyter notebook with all experiments
- `experiment_results.csv` - Results from all 16 experiments
- `data/titanic.db` - Normalized SQLite database
- `data/train.csv` - Original dataset
- `api/app.py` - FastAPI backend code
- `streamlit/app.py` - Streamlit frontend code
- `docker-compose.yml` - Docker deployment configuration
- `models/` - Directory containing all 16 trained models

## Running Locally

To run this project on your local machine:
```bash
git clone https://github.com/ShreyasAravind/housing_app_fall25.git
cd housing_app_fall25
docker-compose up -d
```

Access the application at:
- Streamlit: http://localhost:8501
- API: http://localhost:8000

## Links

- GitHub Repository: https://github.com/ShreyasAravind/housing_app_fall25
- DagsHub Experiments: https://dagshub.com/ShreyasAravind/titanic-classification
- Live Application: http://159.203.81.240:8501

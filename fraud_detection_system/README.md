# 🏦 BFSI Real-Time Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Kafka](https://img.shields.io/badge/Kafka-Streaming-black)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![License](https://img.shields.io/badge/License-MIT-purple)

## 📖 Overview

An end-to-end, production-grade **Machine Learning System** designed for the Banking, Financial Services, and Insurance (BFSI) sector. This system detects fraudulent transactions in real-time using advanced ML techniques, stream processing, and comprehensive monitoring.

It features a modular architecture handling everything from data ingestion to model deployment, specifically tuned for high-class imbalance scenarios typical in fraud detection.

## 🏗️ Architecture

```mermaid
graph LR
    A[Transaction Source] -->|Kafka Stream| B(Stream Processor)
    B -->|Feature Eng.| C{ML Model}
    C -->|Fraud| D[Alert System]
    C -->|Legit| E[Database]
    
    F[Data Lake] -->|Batch| G[Training Pipeline]
    G -->|Artifacts| H[Model Registry]
    H -->|Update| C
    
    I[Monitoring] -->|Drift/Perf| J[Dashboard]


✨ Key Features
⚡ Real-Time Streaming: Kafka-based stream processing for instant fraud scoring.
🧠 Advanced ML Pipeline: Automated feature engineering, handling class imbalance (SMOTE), and hyperparameter tuning (Optuna).
⚖️ BFSI Specifics: Optimized for high Recall and Precision; Cost-sensitive learning.
🛡️ Robust API: FastAPI application for synchronous predictions and model management.
📊 Monitoring & Drift: Integrated Drift Detection (Evidently) and Performance Monitoring.
📦 Dockerized: Fully containerized microservices architecture.
🔄 CI/CD: GitHub Actions pipelines for automated testing and deployment.
🛠️ Tech Stack
Language: Python 3.9
API: FastAPI, Uvicorn
ML Core: Scikit-learn, XGBoost, LightGBM, Imbalanced-learn
Streaming: Apache Kafka, Confluent-Kafka
Orchestration: Docker, Docker Compose
Monitoring: MLflow, Evidently AI, Prometheus (Optional)
Testing: Pytest
🚀 Quick Start
Prerequisites
Docker & Docker Compose
Python 3.9+ (for local dev)




Local Development
Create virtual environment:

Bash

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
Install dependencies:

Bash

pip install -r requirements.txt
Run Training Pipeline:

Bash

python -m src.pipeline.training_pipeline
Start API:

Bash

python app.py
📂 Project Structure
text

fraud_detection_system/
├── .github/workflows/    # CI/CD Pipelines
├── config/               # Configuration files (YAML)
├── docker/               # Dockerfiles
├── src/
│   ├── api/              # FastAPI application
│   ├── components/       # ML Components (Ingestion, Training, etc.)
│   ├── entity/           # Data classes
│   ├── monitoring/       # Drift & Performance Monitoring
│   ├── pipeline/         # Training & Prediction Pipelines
│   └── streaming/        # Kafka Stream Processors
├── tests/                # Unit & Integration Tests
├── app.py                # API Entry point
├── main.py               # Training Entry point
└── stream_app.py         # Streaming Entry point
🔌 API Endpoints
Method	Endpoint	Description
GET	/health	System health check
POST	/api/v1/predict/single	Real-time score for one transaction
POST	/api/v1/predict/batch	Batch scoring
GET	/api/v1/model/info	Current model metadata
📊 Monitoring Dashboard
The system includes a dashboard to monitor:

Data Drift: Detects shifts in transaction patterns.
Model Performance: Tracks Recall/Precision decay over time.
Alerts: System and Fraud alerts log.
🤝 Contributing
Fork the repository
Create a feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request
📄 License
Distributed under the MIT License. See LICENSE for more information.


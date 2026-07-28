# Churn Intelligence Platform: MLOps Pipeline & dbt Transformations Summary

This document summarizes the technical implementations, improvements, and data engineering work completed during our sessions on the **Churn Intelligence Platform** project.

---

## 1. Machine Learning & MLOps Pipeline Upgrades
The machine learning pipeline was upgraded from a standard prediction script into a production-level, version-controlled MLOps pipeline.

* **Data Versioning**: 
  * Integrated dataset tracking using **MLflow Data Versioning** APIs (`mlflow.data`). 
  * Captures the dataset schema and generates a cryptographic hash of the training data to ensure reproducibility.
* **Automated Class Imbalance Resolution**:
  * Implemented balanced sample weight calculations (`Sample Weighting`) within the script to handle the highly skewed class imbalance (representing 96% True Churn rate).
* **Automated Parameter Search (FLAML AutoML & 5-Fold Stratified CV)**:
  * Bound the parameter search space specifically to **LightGBM** to optimize model training.
  * Applied **5-Fold Stratified Cross-Validation** to prevent model overfitting.
  * Resolved internal MLflow autolog mutability conflicts by setting `mlflow_logging=False` inside the FLAML configurations.
* **Manual MLflow Run Tracking & Model Registry**:
  * Configured manual logging for model parameters, run metadata, and evaluation metrics (Accuracy, Macro F1).
  * Programmatically registered the best-performing model to the **MLflow Model Registry** under the name `"Olist_3Class_MLOps_Model"`.
* **Model Serialization Stability**:
  * Adopted `cloudpickle` for model serialization, resolving cross-architecture serialization issues and ensuring stable production deployment.
* **Explainable AI (SHAP Analysis with 3D Slicing)**:
  * Resolved multi-class output incompatibility in SHAP by applying 3D numpy array slicing (`shap_values[..., 2]`) to isolate predictions for Class 2 (True Churn).
  * Generated and saved the SHAP Beeswarm Plot, automatically uploading it as an MLflow run artifact to explain individual customer churn drivers (such as `avg_purchase_gap`).

---

## 2. Data Engineering & dbt Core Transformations
To clean, transform, and structure raw data for model ingestion, a scalable data engineering pipeline was established on **BigQuery** using **dbt Core**.

* **Modular Data Modeling (Staging ➔ Intermediate ➔ Marts)**:
  * Organized raw Olist E-Commerce transactional tables (Orders, Customers, Order Items, Payments, Reviews) into modular, sequential dbt models.
* **SQL Feature Engineering on BigQuery**:
  * Handled raw data cleaning, currency precision, and engineered 15 business-specific features at the database level:
    * *Logistics*: Calculated actual delivery speed and delivery delays against estimates (`delivery_vs_estimated`).
    * *Purchase Behavior*: Calculated repeat purchase intervals and customer-specific buying gaps (`avg_purchase_gap`).
    * *Pricing*: Computed freight-to-price ratios (`freight_ratio`) and payment installment statistics.
* **Downstream Marts for ML Inference**:
  * Created dedicated downstream tables (e.g., `mart_churn_features` and `mart_churn_predictions`) aggregating customer-level feature vectors, preparing clean inputs for the ML training notebook and Streamlit dashboard.
* **Pipeline Scalability & Upgradeability**:
  * Decoupled hardcoded database queries using dbt's modular structure, allowing future schema updates or database migrations to be compiled and run seamlessly using standard dbt commands.

---
*Document prepared for Thanyathorn Krutphan (Earth) — 2026*

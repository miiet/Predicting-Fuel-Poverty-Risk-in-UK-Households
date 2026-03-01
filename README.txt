Predicting Fuel Poverty Risk in UK Households Using Explainable Machine Learning

Project Overview
Fuel poverty is a significant social and economic challenge in the United Kingdom. Households experiencing fuel poverty struggle to afford adequate heating and electricity, often due to a combination of low income, inefficient housing, and high energy costs. Early identification of households at risk allows policymakers and energy providers to prioritise energy-efficiency interventions such as insulation upgrades, heating system improvements, or targeted financial assistance.
This project develops an explainable machine learning pipeline to predict fuel poverty risk using housing characteristics, socio-economic indicators, and energy-efficiency information. The objective is to create a decision-support tool capable of identifying households that may require further assessment or policy intervention.

The project emphasises three critical aspects of responsible machine learning:
Data leakage prevention to ensure reliable and realistic model evaluation
Model transparency through explainable AI techniques (SHAP)
Policy-oriented interpretation of model outputs

The final solution demonstrates a full applied machine learning workflow from data preprocessing to explainable model insights, suitable for public policy analytics and energy-sector decision support.

Dataset
The dataset used in this project is derived from an anonymised sample of the National Energy Efficiency Data Framework (NEED).
The NEED dataset combines information from multiple sources including property records, energy efficiency measures, and regional indicators.
The dataset contains approximately 50,000 household records and includes features such as:
Property Characteristics
Property type (Detached, Semi-detached, Flat, etc.)
Property age band
Floor area band
Conservatory indicator
Socioeconomic Indicators
Council tax band
Index of Multiple Deprivation (IMD) band
Regional Information
Region codes representing UK geographical areas
Energy Efficiency Indicators
Energy Performance Certificate (EPC)
Loft insulation flag
Cavity wall insulation flag
Solar photovoltaic installation flag
Heating System
Main heating fuel type

The target variable used for modelling is:

fuel_poverty_risk

This is a binary variable indicating whether a household is considered at risk of fuel poverty.

Data Leakage Handling
During early experimentation, the dataset included features representing recent gas and electricity consumption averages:

Gcons_recent_avg
Econs_recent_avg

Although these features appear useful, they introduce data leakage because energy consumption directly reflects household heating behaviour and energy affordability. Using these features would artificially inflate model performance by indirectly revealing the outcome.

To ensure a realistic modelling approach, these variables were removed from the final training pipeline. All reported model results therefore use a no-leakage feature set, ensuring that predictions are based only on structural and contextual information available before the outcome occurs.

Machine Learning Models

Several classification models were implemented and compared in order to evaluate different algorithmic approaches.

The models trained in this project include:

Logistic Regression (baseline model)

LightGBM

XGBoost

CatBoost

Each model was trained using the same cleaned dataset and evaluated on a held-out test set. Class imbalance was handled using class weighting or model-specific imbalance handling methods.

Model evaluation included the following metrics:

ROC-AUC score

Precision

Recall

F1 score

Confusion matrix

Threshold analysis

The final models achieved approximately:

ROC-AUC ≈ 0.94 – 0.95

The models were tuned to prioritise high recall for the fuel poverty risk class, which is appropriate for a screening application where missing vulnerable households should be minimised.

Model Explainability (SHAP)

To ensure transparency and interpretability, the project uses SHAP (SHapley Additive exPlanations).

SHAP allows us to understand:

Which features contribute most to model predictions

How feature values increase or decrease predicted risk

Why the model made a particular prediction for an individual household

For each gradient boosting model, the following explainability visualisations were generated:

Global Feature Importance

A bar plot ranking features based on their overall contribution to model predictions.

SHAP Summary (Beeswarm Plot)

A distribution plot showing how high and low values of each feature influence the model output across the dataset.

SHAP Waterfall Plot

A local explanation showing how individual feature contributions combine to produce a prediction for a specific household.

These explanations help translate model outputs into policy-relevant insights, allowing analysts to understand the structural drivers of predicted fuel poverty risk.

Project Structure
fuel_poverty_ml
│
├── data
│   └── processed
│       ├── need_train.csv
│       └── need_test.csv
│
├── outputs
│   ├── figures
│   ├── metrics
│   ├── models
│   ├── shap_lightgbm_no_leakage
│   ├── shap_xgboost_no_leakage
│   └── shap_catboost_no_leakage
│
├── src
│   ├── data processing scripts
│   ├── model training scripts
│   ├── evaluation scripts
│   └── SHAP explainability scripts
│
└── README.md

The src folder contains modular scripts for each stage of the machine learning pipeline including preprocessing, model training, evaluation, and explainability.

The outputs folder contains saved models, metrics, ROC curves, and SHAP visualisations generated during model execution.

How to Run the Project

To reproduce the results locally, follow the steps below.

1. Clone the Repository
git clone https://github.com/your-username/fuel-poverty-ml.git
cd fuel-poverty-ml
2. Create a Python Virtual Environment
python -m venv .venv

Activate the environment:

Windows

.venv\Scripts\activate

Mac/Linux

source .venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
Training the Models

To train and evaluate the primary models, run the following scripts.

LightGBM model
python src/16_save_lightgbm_no_leakage_results.py
XGBoost model
python src/18_xgboost_no_leakage.py
CatBoost model
python src/19_catboost_no_leakage.py

These scripts will generate:

Model performance metrics

ROC curves

Saved model files

All outputs will be stored inside the outputs directory.

Generating SHAP Explanations

To generate model interpretability plots, run:

XGBoost SHAP explanations
python src/21_shap_xgboost_no_leakage.py
CatBoost SHAP explanations
python src/22_shap_catboost_no_leakage.py

These scripts generate SHAP visualisations and save them to:

outputs/shap_xgboost_no_leakage
outputs/shap_catboost_no_leakage
Checking the Outputs

If you are reviewing the repository without re-running the scripts, you can directly inspect the results in the outputs folder.

Key files include:

Model Metrics

outputs/metrics/

Saved Models

outputs/models/

ROC Curve Visualisations

outputs/figures/

SHAP Explainability Plots

outputs/shap_lightgbm_no_leakage/
outputs/shap_xgboost_no_leakage/
outputs/shap_catboost_no_leakage/

These outputs demonstrate the trained models, evaluation metrics, and explainability analysis generated during the project.

Ethical Considerations

This model is intended strictly for decision support.

Predictions should not be used to automatically determine eligibility for assistance or policy interventions. Instead, the model can help analysts prioritise households for further review, enabling more efficient allocation of resources.

Additionally, the dataset does not contain direct measures of household income or vulnerability. As a result, the model relies on structural proxies such as housing characteristics and regional deprivation indicators.

Any real-world deployment would require further validation, fairness analysis, and integration with broader policy frameworks.

Conclusion

This project demonstrates a complete and responsible machine learning workflow, including data preparation, leakage-aware modelling, robust evaluation, and explainable AI.

By combining gradient boosting models with SHAP-based interpretability, the project illustrates how machine learning can support transparent, policy-relevant insights into complex social issues such as fuel poverty.

The repository is designed to be fully reproducible, allowing others to inspect the code, run the models, and explore the resulting explanations.
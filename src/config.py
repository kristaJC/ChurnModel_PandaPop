### MLFLOW Experiment
EXPERIMENT_NAME = "/Users/krista@jamcity.com/PP-Churn-Model"


# TARGET LABEL
LABEL_COL = "churn7"


### Tables 

# Source data for all features
FEATURES_TABLE_NAME = "teams.data_science.pp_churn_features"


# Destination table for predictions from model
PREDICTION_TABLE_NAME = 'teams.data_science.pp_churn_predictions'


# Destination table for real churn labels 
LABEL_TABLE_NAME = 'teams.data_science.pp_churn_actuals'
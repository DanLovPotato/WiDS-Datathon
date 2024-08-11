import os
import json
import warnings
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pycox.models import CoxPH, DeepHitSingle
import torchtuples as tt

# Set up warnings
warnings.filterwarnings("ignore")

# Set up file paths
path_dir = os.path.dirname(os.getcwd())
path_models = os.path.join(path_dir, "DATATHON", "outputs", "models")

# Define column names
cols_x = ['patient_zip3', 'patient_age', 'population', 'density', 
          'age_median', 'age_under_10', 'age_10_to_19', 'age_20s', 
          'age_30s', 'age_40s', 'age_50s', 'age_60s', 'age_70s', 
          'age_over_80', 'male', 'female', 'married', 'divorced', 
          'never_married', 'widowed', 'family_size', 'home_ownership', 
          'housing_units', 'home_value', 'rent_median', 'rent_burden', 
          'education_less_highschool', 'education_highschool', 'education_some_college', 
          'education_bachelors', 'education_graduate', 'education_college_or_above', 
          'education_stem_degree', 'labor_force_participation', 'unemployment_rate', 
          'self_employed', 'farmer', 'race_white', 'race_black', 'race_asian', 'race_native', 
          'race_pacific', 'race_other', 'race_multiple', 'hispanic', 'disabled', 'poverty', 
          'limited_english', 'commute_time', 'health_uninsured', 'veteran',
          'days_of_extreme_heat', 'days_of_extreme_cold', 'temp_std_dev', 'temp_range', 'winter_avg_2013', 
          'winter_avg_2014', 'winter_avg_2015', 'winter_avg_2016', 'winter_avg_2017', 'winter_avg_2018', 
          'summer_avg_2013', 'summer_avg_2014', 'summer_avg_2015', 'summer_avg_2016', 'summer_avg_2017', 
          'summer_avg_2018', 'fall_avg_2013', 'fall_avg_2014', 'fall_avg_2015', 'fall_avg_2016', 
          'fall_avg_2017', 'fall_avg_2018', 'spring_avg_2013', 'spring_avg_2014', 'spring_avg_2015', 
          'spring_avg_2016', 'spring_avg_2017', 'spring_avg_2018', 'temp_jan', 'temp_feb', 'temp_mar', 
          'temp_apr', 'temp_may', 'temp_jun', 'temp_jul', 'temp_aug', 'temp_sep', 'temp_oct', 'temp_nov', 
          'temp_dec', 'temp_13', 'temp_14', 'temp_15', 'temp_16', 'temp_17', 'temp_18', 'income_35_below', 
          'income_35_75', 'income_75_above', 'Max AQI(avbystate)', '90th Percentile AQI(avbystate)', 
          'Median AQI(avbystate)', 'disaster_num_2013', 'disaster_num_2014', 'disaster_num_2015', 
          'disaster_num_2016', 'disaster_num_2017', 'disaster_num_2018', 'RISK_SCORE', 
          'breast_cancer_diagnosis_code_isIcd', 'temp_change', 'yearly_temp_increase',
          'patient_race_black', 'patient_race_hispanic', 'patient_race_other', 'patient_race_unknown', 
          'patient_race_white', 'payer_type_medicaid', 'payer_type_medicare advantage', 'payer_type_unknown', 
          'state_al', 'state_ar', 'state_az', 'state_ca', 'state_co', 'state_dc', 'state_de', 'state_fl', 
          'state_ga', 'state_hi', 'state_ia', 'state_id', 'state_il', 'state_in', 'state_ks', 'state_ky', 
          'state_la', 'state_md', 'state_mi', 'state_mn', 'state_mo', 'state_ms', 'state_mt', 'state_nc', 
          'state_nd', 'state_ne', 'state_nm', 'state_nv', 'state_ny', 'state_oh', 'state_ok', 'state_or', 
          'state_pa', 'state_sc', 'state_sd', 'state_tn', 'state_tx', 'state_ut', 'state_va', 'state_wa', 
          'state_wi', 'state_wv', 'state_wy', 'Region_northeast', 'Region_south', 'Region_west', 
          'Division_east south central', 'Division_middle atlantic', 'Division_mountain', 'Division_pacific', 
          'Division_south atlantic', 'Division_west north central', 'Division_west south central', 
          'metastatic_cancer_diagnosis_code_c771', 'metastatic_cancer_diagnosis_code_c772', 
          'metastatic_cancer_diagnosis_code_c773', 'metastatic_cancer_diagnosis_code_c774', 
          'metastatic_cancer_diagnosis_code_c775', 'metastatic_cancer_diagnosis_code_c778', 
          'metastatic_cancer_diagnosis_code_c779', 'metastatic_cancer_diagnosis_code_c7800', 
          'metastatic_cancer_diagnosis_code_c7801', 'metastatic_cancer_diagnosis_code_c7802', 
          'metastatic_cancer_diagnosis_code_c781', 'metastatic_cancer_diagnosis_code_c782', 
          'metastatic_cancer_diagnosis_code_c7839', 'metastatic_cancer_diagnosis_code_c784', 
          'metastatic_cancer_diagnosis_code_c785', 'metastatic_cancer_diagnosis_code_c786', 
          'metastatic_cancer_diagnosis_code_c787', 'metastatic_cancer_diagnosis_code_c7880', 
          'metastatic_cancer_diagnosis_code_c7889', 'metastatic_cancer_diagnosis_code_c7900', 
          'metastatic_cancer_diagnosis_code_c7901', 'metastatic_cancer_diagnosis_code_c7910', 
          'metastatic_cancer_diagnosis_code_c7911', 'metastatic_cancer_diagnosis_code_c792', 
          'metastatic_cancer_diagnosis_code_c7931', 'metastatic_cancer_diagnosis_code_c7932', 
          'metastatic_cancer_diagnosis_code_c7940', 'metastatic_cancer_diagnosis_code_c7949', 
          'metastatic_cancer_diagnosis_code_c795', 'metastatic_cancer_diagnosis_code_c7951', 
          'metastatic_cancer_diagnosis_code_c7952', 'metastatic_cancer_diagnosis_code_c7960', 
          'metastatic_cancer_diagnosis_code_c7961', 'metastatic_cancer_diagnosis_code_c7962', 
          'metastatic_cancer_diagnosis_code_c7970', 'metastatic_cancer_diagnosis_code_c7971', 
          'metastatic_cancer_diagnosis_code_c7972', 'metastatic_cancer_diagnosis_code_c798', 
          'metastatic_cancer_diagnosis_code_c7981', 'metastatic_cancer_diagnosis_code_c7982', 
          'metastatic_cancer_diagnosis_code_c7989', 'metastatic_cancer_diagnosis_code_c799', 
          'breast_cancer_diagnosis_icd10_c50011', 'breast_cancer_diagnosis_icd10_c50012', 
          'breast_cancer_diagnosis_icd10_c50019', 'breast_cancer_diagnosis_icd10_c5011', 
          'breast_cancer_diagnosis_icd10_c50111', 'breast_cancer_diagnosis_icd10_c50112', 
          'breast_cancer_diagnosis_icd10_c50119', 'breast_cancer_diagnosis_icd10_c50122', 
          'breast_cancer_diagnosis_icd10_c50211', 'breast_cancer_diagnosis_icd10_c50212', 
          'breast_cancer_diagnosis_icd10_c50219', 'breast_cancer_diagnosis_icd10_c50221', 
          'breast_cancer_diagnosis_icd10_c5031', 'breast_cancer_diagnosis_icd10_c50311', 
          'breast_cancer_diagnosis_icd10_c50312', 'breast_cancer_diagnosis_icd10_c50319', 
          'breast_cancer_diagnosis_icd10_c5041', 'breast_cancer_diagnosis_icd10_c50411', 
          'breast_cancer_diagnosis_icd10_c50412', 'breast_cancer_diagnosis_icd10_c50419', 
          'breast_cancer_diagnosis_icd10_c50421', 'breast_cancer_diagnosis_icd10_c50511', 
          'breast_cancer_diagnosis_icd10_c50512', 'breast_cancer_diagnosis_icd10_c50519', 
          'breast_cancer_diagnosis_icd10_c50611', 'breast_cancer_diagnosis_icd10_c50612', 
          'breast_cancer_diagnosis_icd10_c50619', 'breast_cancer_diagnosis_icd10_c5081', 
          'breast_cancer_diagnosis_icd10_c50811', 'breast_cancer_diagnosis_icd10_c50812', 
          'breast_cancer_diagnosis_icd10_c50819', 'breast_cancer_diagnosis_icd10_c509', 
          'breast_cancer_diagnosis_icd10_c5091', 'breast_cancer_diagnosis_icd10_c50911', 
          'breast_cancer_diagnosis_icd10_c50912', 'breast_cancer_diagnosis_icd10_c50919', 
          'breast_cancer_diagnosis_icd10_c50922', 'breast_cancer_diagnosis_icd10_c7981', 
          'metastatic_cancer_diagnosis_code_c7830', 'metastatic_cancer_diagnosis_code_c7919', 
          'breast_cancer_diagnosis_icd10_c5021'
          ]

col_target = 'metastatic_diagnosis_period'

# Read the test data
test_df = pd.read_csv(os.path.join(path_dir, "DATATHON", "data", "final_test_df.csv"))
X_test = pd.read_csv(os.path.join(path_dir, "DATATHON", "outputs", "data", "test_data_scaled.csv"))

# Read the solution template
solution_df = pd.read_csv(os.path.join(path_dir, "DATATHON", "data", "external", "solution_template.csv"))

# Function to load the model
def load_model(filename, model_obj, in_features, out_features, params):
    num_nodes = [int(params["n_nodes"])] * (int(params["n_layers"]))
    del params["n_nodes"]
    del params["n_layers"]

    if 'model_params' in params.keys():
        model_params = json.loads(params['model_params'].replace('\'', '\"'))
        del params['model_params']
        net = tt.practical.MLPVanilla(
            in_features=in_features, out_features=out_features, num_nodes=num_nodes, **params)
        model = model_obj(net, **model_params)
    else:
        net = tt.practical.MLPVanilla(
            in_features=in_features, out_features=out_features, num_nodes=num_nodes, **params)
        model = model_obj(net)
    model.load_net(os.path.join(path_models, filename))

    return model

# Function to generate predictions
def generate_predictions(model_name):
    global solution_df
    # Load the model
    files_model = [p for p in os.listdir(path_models) if '.pt' in p]
    models = {}

    if model_name not in ['deepsurv', 'deephit']:
        print("Invalid model name. Available models: deepsurv, deephit")
        return

    table_final = pd.read_csv(os.path.join(path_dir, "DATATHON", "outputs", "model_scores_dl.csv"))

    if model_name == 'deepsurv':
        params = table_final[table_final['model'] == "DeepSurv"].dropna(axis=1) \
            .drop(['model', 'lr'] + [c for c in table_final.columns if 'score' in c], axis=1) \
            .iloc[0].to_dict()
        models['deepsurv'] = load_model("deepsurv.pt", CoxPH, len(cols_x), 1, params)
    elif model_name == 'deephit':
        params = table_final[table_final['model'] == "DeepHit"].dropna(axis=1) \
            .drop(['model', 'lr', 'batch_size', 'discrete'] + [c for c in table_final.columns if 'score' in c],
                  axis=1) \
            .iloc[0].to_dict()
        models['deephit'] = load_model("deephit.pt", DeepHitSingle, len(cols_x), 1, params)

    model = models[model_name]
    survs = model.predict_surv_df(np.array(X_test[cols_x]).astype(np.float32))

    def median_survival_time(probabilities, days):
        median_times = {}
        threshold = 0.5

        for column in probabilities.columns:
            survival_probs = probabilities[column]
            below_threshold = survival_probs < threshold
            if np.any(below_threshold):
                first_below = np.where(below_threshold)[0][0]
                median_times[column] = days[first_below]
            else:
                median_times[column] = days[-1] + 1

        return median_times

    days = np.arange(366)
    # print(survs)
    median_survival_times = median_survival_time(survs, days)

    median_times_df = pd.DataFrame(list(median_survival_times.items()), columns=['Individual', 'metastatic_diagnosis_period'])

    solution = pd.concat([test_df[['patient_id']], median_times_df[['metastatic_diagnosis_period']]], axis=1)
    solution_df = solution_df.merge(solution, left_on='patient_id', right_on='patient_id') \
        .drop(columns=['metastatic_diagnosis_period_x']).rename(columns={"metastatic_diagnosis_period_y": "metastatic_diagnosis_period"})

    solution_df.to_csv("./outputs/data/predictions.csv", index=False)
    print("Predictions generated successfully")

# Chnage the model name to 'deepsurv' or 'deephit'
generate_predictions('deepsurv')









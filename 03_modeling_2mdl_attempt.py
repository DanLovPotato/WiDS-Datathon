
import pandas as pd
import numpy as np
import os
from backend import *
from scipy.sparse import csc_matrix
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.compose import ColumnTransformer
from pycaret.containers.models.regression import RegressorContainer
from sklearn.metrics import mean_squared_error
from lifelines import CoxPHFitter
import math
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

from pycaret.classification import setup as clf_setup, create_model as create_clf_model, tune_model as tune_clf_model, compare_models as compare_clf_models, predict_model as predict_clf_model, plot_model as plot_clf_model, add_metric as add_clf_metric
from pycaret.regression import setup as reg_setup, create_model as create_reg_model, tune_model as tune_reg_model, compare_models as compare_reg_models, predict_model as predict_reg_model, plot_model as plot_reg_model



data_folder = "data"
external_folder = "external"
train_file_name = "final_train_df.csv"
test_file_name = "final_test_df.csv"
solution_file_name = 'solution_template.csv'

train_file = os.path.join(data_folder, train_file_name)
test_file = os.path.join(data_folder, test_file_name)
solution_file = os.path.join(data_folder, external_folder, solution_file_name)

# Load your datasets
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
solution_df = pd.read_csv(solution_file)


def preprocessing(df):

  # Translate breast_cancer_diagnosis_icd10 and metastatic_cancer_diagnosis_code
  df = get_diagnosis_cat_cols(df)
  col_02 = [#'breast_cancer_diagnosis_icd10',
            #'breast_cancer_diagnosis_code',
            'breast_cancer_diagnosis_desc',
            #'breast_cancer_type', 'breast_cancer_affected_location',
            #'metastatic_cancer_diagnosis_code', 'metastatic_cancer_affected_location',
            #'breast_cancer_diagnosis_code_isIcd' 
            ]
  
  # Handle breast_cancer_diagnosis_code_isICD
  df['breast_cancer_diagnosis_code_isIcd'] = df['breast_cancer_diagnosis_code_isIcd'].apply(lambda x: 1 if x == True else 0)

 # Rename Col Name
  df.rename(columns={"Max AQI(avbystate)": "aqi_max", "90th Percentile AQI(avbystate)": "aqi_90_pctl", "Median AQI(avbystate)": "aqi_median"}, errors="raise", inplace = True)

  # aqi extremes handling 
  # 'aqi_median', 'aqi_90_pctl', 'aqi_max'
  df['aqi_median_is_outlier'] = df['aqi_median'].apply(lambda x: 1 if x > 45 else 0)
  df['aqi_median'] = df['aqi_median'].clip(upper=45)

  df['aqi_90_pctl_is_outlier'] = df['aqi_90_pctl'].apply(lambda x: 1 if x > 65 else 0)
  df['aqi_90_pctl'] = df['aqi_90_pctl'].clip(upper=65)

  df['aqi_max_is_outlier'] = df['aqi_max'].apply(lambda x: 1 if x > 130 else 0)
  df['aqi_max'] = df['aqi_max'].clip(upper=130)

  # extreme weather handling
  df['has_extreme_cold_days'] = pd.cut(x=df['days_of_extreme_cold'], bins=[-1,0,10,30,366],labels=['None','1-10','11-30','31+'])
  df['has_extreme_hot_days'] = df['days_of_extreme_heat'].apply(lambda x: 1 if x > 0 else 0)
  col_06 = ['days_of_extreme_cold', 'days_of_extreme_heat']
  
  # handling 'farmer'
  df['no_farmer'] = df['farmer'].apply(lambda x: 1 if x == 0 else 0)
  df['farmer'] = df['farmer'] + 0.000001  # add very small abritrary value to help with transformation


  # Handle Zip3
  df = get_zip_freq_pctle(df)
  #df['patient_zip3'] = df['patient_zip3'].astype('category')
  #df['patient_zip2'] = df['patient_zip3'].apply(lambda x: x[:3]).astype('category')
  col_05 = ['patient_zip3']

  # Combine race
  df['race_other'] = df['race_native']+df['race_pacific']+df['race_other']+df['race_multiple']
  col_03 = [#'race_native',	'race_pacific',	'race_other',	'race_multiple'
  ]

  # Categorize Age
  df['age_group'] = pd.cut(x=df['patient_age'], bins=[-1,20,40,50,60,80,100],labels=['0-20','21-40','41-50','51-60','61-80','81-100'])
  col_04 = [
      #'patient_age'
  ]

  # Mean-ing disaster_num
  #df['ICD9_disaster_num'] = (df['disaster_num_2013'] + df['disaster_num_2014'])/2
  #df['ICD10_disaster_num'] = (df['disaster_num_2015'] + df['disaster_num_2016']+df['disaster_num_2017'] + df['disaster_num_2018'])/4
  col_01 = ['disaster_num_2013','disaster_num_2014','disaster_num_2015','disaster_num_2016','disaster_num_2017','disaster_num_2018'
      ]

 

  # Drop Features
  drop_col = ['patient_id',"patient_gender", 
              'Region', 'Division', 
              'age_under_10',	'age_10_to_19',	'age_20s',	'age_30s',	'age_60s',	'age_70s',	'age_over_80', 'male',
              'education_highschool','education_some_college','education_bachelors','education_graduate',
              #'tempature_jan','tempature_feb','tempature_mar','tempature_apr','tempature_may','tempature_jun',
              #'tempature_jul','tempature_aug','tempature_sep','tempature_oct','tempature_nov','tempature_dec'
              ]
  [drop_col.extend(l) for l in [col_03, col_04, col_01,col_02, col_05, col_06]]
  
  # get train_y col
  if 'metastatic_diagnosis_period' in df.columns:
      drop_col.extend(['metastatic_diagnosis_period'])

  X = df.drop(columns=drop_col)
  
  return X


def get_diagnosis_cat_cols(df):

    # breast_cancer_diagnosis_code lookup path
    bcd = os.path.join(data_folder, external_folder, 'breast_cancer_lookup.csv')
    bcd_df = pd.read_csv(bcd)
    
    # metastatic_cancer_diagnosis_code lookup path
    mcd = os.path.join(data_folder, external_folder, 'metastatic_cancer_lookup.csv')
    mcd_df = pd.read_csv(mcd)  

    df = df.merge(bcd_df, on='breast_cancer_diagnosis_icd10', how='left')
    df = df.merge(mcd_df, on='metastatic_cancer_diagnosis_code', how='left')

    return df

def get_zip_freq_pctle(df):

    # patient_zip3 lookup path
    pctle = os.path.join(data_folder, external_folder, 'zip3_freq_pct_lookup.csv')
    pctle_df = pd.read_csv(pctle)

    df = df.merge(pctle_df, on='patient_zip3', how='left')

    return df


def normalizing(X, 
                dummy_items = ['no_farmer', 'breast_cancer_diagnosis_code_isIcd'
                               ,'aqi_median_is_outlier','aqi_90_pctl_is_outlier','aqi_max_is_outlier'
                               , 'has_extreme_hot_days']):

  categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
  numeric_columns = X.select_dtypes(exclude=['object', 'category']).columns.tolist()

  for i in dummy_items:
    numeric_columns.remove(i)

  power_columnes = [#'yearly_temp_increase', 'temp_change'
                    #,'self_employed', 'unemployment_rate', 
                    #,'labor_force_participation', 'education_stem_degree'
                    #,'rent_burden'
                    #, 'female'

  ]

  for i in power_columnes:
      numeric_columns.remove(i)


  
  encoder = OneHotEncoder(sparse_output=False)
  one_hot_encoded = encoder.fit_transform(X[categorical_columns])
  one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))  


  # # Power Transform
  # stdscaler = StandardScaler()
  # stdscaler_encoded = stdscaler.fit_transform(X[power_columnes]) 
  # stdscaler_df = pd.DataFrame(stdscaler_encoded, columns=stdscaler.get_feature_names_out(power_columnes))  
  
  # transformer = PowerTransformer()
  # transform_encoded = transformer.fit_transform(stdscaler_df)
  # transform_df = pd.DataFrame(transform_encoded, columns=transformer.get_feature_names_out(power_columnes))  

  # Quantile Transform

  q_transformer = QuantileTransformer(random_state=42,output_distribution='normal')
  q_transform_encoded = q_transformer.fit_transform(X[numeric_columns]) 
  q_transform_df = pd.DataFrame(q_transform_encoded, columns=q_transformer.get_feature_names_out(numeric_columns))  


#   if y is not None:
#       df_encoded = pd.concat([one_hot_df, transform_df, y], axis=1)
#       df_encoded['duration'] = df_encoded['metastatic_diagnosis_period']
#       df_encoded['event'] = 1
#   else:
  
  df_encoded = pd.concat([one_hot_df
                          #, transform_df
                          ,q_transform_df], axis=1)

  return df_encoded

# Prepare datasets
preprocessed_train_df = preprocessing(train_df)
preprocessed_train_df.to_csv("./data/preprocessed_train_df.csv", index=False)
preprocessed_test_df = preprocessing(test_df)
preprocessed_test_df.to_csv("./data/preprocessed_test_df.csv", index=False)

X_train = normalizing(preprocessed_train_df)
X_test = normalizing(preprocessed_test_df)

missingcol = [i for i in X_train.columns.tolist() if i not in X_test.columns.tolist()]
X_test[missingcol] = 0

missingcol = [i for i in X_test.columns.tolist() if i not in X_train.columns.tolist()]
X_train[missingcol] = 0

# Reorder the columns of the test DataFrame to match the order of the train DataFrame
X_test = X_test[X_train.columns]

X_test['summer_avg_2018'] = X_test['summer_avg_2018'].fillna(78.57) #ga avg
X_test['fall_avg_2018'] = X_test['fall_avg_2018'].fillna(53.55) #ny avg

X_train.to_csv("./data/normalized_train_df.csv", index=False)
X_test.to_csv("./data/normalized_test_df.csv", index=False)

categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

y = train_df['metastatic_diagnosis_period']

# PCA
pca = PCA(n_components=44)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


threshold = 0.5  # Adjust this threshold based on your needs

def adjusted_recall(y_true, y_pred, threshold=0.5):
    from sklearn.metrics import recall_score
    
    # Adjust predictions based on the threshold
    y_pred_adjusted = (y_pred[:, 1] > threshold).astype(int)
    
    return recall_score(y_true, y_pred_adjusted)


# Setup PyCaret for classification models
# predict if the number of days is 0 or not
y_bin = (y > 0).astype(int)
clf = clf_setup(data=X_train_pca, 
            target=y_bin, 
            session_id=123,
            normalize=True,
            imputation_type='simple',
            fix_imbalance=True,  # Enable imbalance handling
            fix_imbalance_method='smote') # fix_imbalance_method='undersampling'

#add_clf_metric('adjusted_recall', 'Adjusted Recall', adjusted_recall, greater_is_better=True)

# Comment out inefficient models, leaving CatBoostfor training

# # AdaBoost
# print('CLF - Tuning Ada Boost')
# adaboost = create_model('ada')
# adaboost_tuned = tune_model(adaboost)

# # Gradient Boosting
# print('CLF - Tuning Gradient Boost')
# gradient_boost = create_model('gbc')
# gradient_boost_tuned = tune_model(gradient_boost)

# # Light gbm
# print('CLF - Tuning Light GBM')
# lightgbm = create_model('lightgbm')
# lightgbm_tuned = tune_model(lightgbm)

# CatBoost 
# print('CLF - Creating Cat Boost')
# catboost = create_clf_model('catboost')
# print('CLF - Tuning Cat Boost')
# catboost_tuned_acc = tune_clf_model(catboost, custom_scorer='adjusted_recall') # Adjusted weights  # optimize='Recall' 

# # Compare the ensemble models
# print('CLF - Ensembling models')
# ensemble_models = [catboost
#                     #catboost_tuned_acc, 
#                     #adaboost_tuned, gradient_boost_tuned, lightgbm_tuned
#                         ]
# print('CLF - Selecting model')
# best_ensemble_model_clf = compare_clf_models(ensemble_models, sort='adjusted_recall') #sort='Recall'
best_ensemble_model_clf = compare_clf_models(sort='recall')

plot_clf_model(best_ensemble_model_clf, plot='feature')  # Feature Importance
plot_clf_model(best_ensemble_model_clf, plot='confusion_matrix')   # Error Analysis


# Setup PyCaret for regression models

# Filter X and y with non-zero metastatic_diagnosis_period
non_zero_idx = y > 0
X_train_pca = pd.DataFrame(X_train_pca)
X_reg = X_train_pca.loc[non_zero_idx]
y_reg = y.loc[non_zero_idx].astype(float)

reg = reg_setup(data=X_reg,
            train_size = 0.7,
            target=y_reg, 
            index=False,
            session_id=123, #random_seed
            #     ignore_low_variance = True,
            remove_multicollinearity = True,
            multicollinearity_threshold = 0.95
            )             

# # CatBoost 
# print('REG - Creating Cat Boost')
# catboost_RMSE = create_reg_model('catboost')
# print('REG - Tuning Cat Boost')
# catboost_tuned_RMSE = tune_reg_model(catboost_RMSE, optimize='RMSE')

# lightgbm_reg = create_model('lightgbm')
# lightgbm_reg_tuned = tune_model(lightgbm_reg, optimize='RMSE')

# xgboost_reg = create_model('xgboost')
# xgboost_reg_tuned = tune_model(xgboost_reg, optimize='RMSE')


try:
    best_model_reg = compare_reg_models(#include=[#'catboost', 'lightgbm', 'xgboost', 
                                         #lightgbm_reg_tuned, xgboost_reg_tuned, 
                                         #catboost_tuned_RMSE],
                                        sort="RMSE")
except Exception as e:
    print(f"Error during model comparison: {e}")

plot_reg_model(best_model_reg, plot='feature')  # Feature Importance
plot_reg_model(best_model_reg, plot='error')   # Error Analysis
plot_reg_model(best_model_reg, plot='residuals')   # Residual Analysis
#plot_reg_model(best_model_reg, plot='feature_all')  # Feature Importance

# Prediction 

# 1. Use the classification model to predict if the number of days is 0 or not
clf_prediction = predict_clf_model(best_ensemble_model_clf, data=X_test_pca, raw_score=True)

# Adjust threshold

clf_prediction['adjusted_label'] = (clf_prediction['prediction_score_1'] > threshold).astype(int)


# 2: If the classification model predicts non-zero, use the regression model to predict the number of days

X_test_pca['metastatic_diagnosis_period'] = 0
non_zero_mask = clf_prediction['adjusted_label'] == 1


# Predict with the regression model for non-zero cases
reg_prediction = predict_reg_model(best_model_reg, data=X_test_pca.loc[non_zero_mask])
reg_prediction['orig_index'] = non_zero_mask[non_zero_mask].index

# Assign regression predictions to the appropriate rows in the original dataset
X_test_pca['metastatic_diagnosis_period'] = X_test_pca.index.map(dict(zip(reg_prediction['orig_index'], reg_prediction['prediction_label']))).fillna(0)

# 3: Correcting value to within range of 0 to 365 
X_test_pca['metastatic_diagnosis_period'] = np.round(X_test_pca['metastatic_diagnosis_period']).astype(int)
X_test_pca['metastatic_diagnosis_period'] = np.clip(X_test_pca['metastatic_diagnosis_period'], 0, 365)

# 4: output solution
solution_df = pd.concat([test_df[['patient_id']], X_test_pca[['metastatic_diagnosis_period']]], axis=1)
solution_df.to_csv("./data/predictions.csv", index=False)


import plotly.express as px
import plotly.graph_objs as go
#plot PCA
pca_df = pd.DataFrame(data=X_train_pca[:, :2], columns=['PC1', 'PC2'])
pca_df['target'] = y

# Explained variance
explained_variance = pca.explained_variance_ratio_

# Scatter plot of the observations in the PCA space
fig = px.scatter(pca_df, x='PC1', y='PC2', color='target', 
                 title=f'PCA Scatter Plot (explained variance: PC1={explained_variance[0]:.2f}, PC2={explained_variance[1]:.2f})')

# Add vectors for the features
# The principal components are already scaled by the singular values (sqrt of explained variance)
for i, feature in enumerate(train_df.columns):
    fig.add_trace(go.Scatter(
        x=[0, pca.components_[0, i] * np.sqrt(pca.explained_variance_[0])],
        y=[0, pca.components_[1, i] * np.sqrt(pca.explained_variance_[1])],
        mode='lines+text',
        text=[None, feature],
        textposition='top center',
        name=feature
    ))

# Show the plot
fig.show()




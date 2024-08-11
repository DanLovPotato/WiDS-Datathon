
import pandas as pd
import numpy as np
import os
from backend import *

from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler, MinMaxScaler, QuantileTransformer
import math
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
  col_02 = ['breast_cancer_diagnosis_desc'
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
  col_06 = ['days_of_extreme_cold', 'days_of_extreme_heat'
            ,'aqi_median', 'aqi_90_pctl', 'aqi_max'
            ]
  
  
  # handling 'farmer'
  df['no_farmer'] = df['farmer'].apply(lambda x: 1 if x == 0 else 0)
  df['farmer'] = df['farmer'] + 0.000001  # add very small abritrary value to help with transformation

  # Handle Zip3
  df = get_zip_freq_pctle(df)
  col_05 = [#'patient_zip3'
     ]

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
  drop_col = ['patient_id',"patient_gender", 'male',
              #'Region', 'Division', 
              #'age_under_10',	'age_10_to_19',	'age_20s',	'age_30s',	'age_60s',	'age_70s',	'age_over_80', 
              #'education_highschool','education_some_college','education_bachelors','education_graduate',
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

def normalizing(X, test=False,
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

  if test:
    # One Hot Encoding
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(X[categorical_columns])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))  
  else:
    # Dummy Encoding
    one_hot_df = pd.DataFrame()
    for i in categorical_columns: 
      dummy_df = pd.get_dummies(X[i], drop_first=True, prefix=i, dummy_na=True)
      one_hot_df = pd.concat([one_hot_df, dummy_df], axis=1)
    
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

  df_encoded = pd.concat([one_hot_df
                          #, transform_df
                          ,q_transform_df], axis=1)

  return df_encoded

# Prepare datasets
preprocessed_train_df = preprocessing(train_df)
preprocessed_train_df.to_csv("./data/preprocessed_train_df.csv", index=False)
preprocessed_test_df = preprocessing(test_df)
preprocessed_test_df.to_csv("./data/preprocessed_test_df.csv", index=False)

X_train = normalizing(preprocessed_train_df, test=False)
X_test = normalizing(preprocessed_test_df, test=True)

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

# modeling
reg = reg_setup(data=X_train,
            train_size = 0.7,
            target=y, 
            index=False,
            session_id=123, #random_seed
            #     ignore_low_variance = True,
            remove_multicollinearity = True,
            multicollinearity_threshold = 0.95
            )             

print('REG - Creating gbr Boost')
gbr = create_reg_model('gbr')
print('REG - Tuning gbr Boost')
gbr_tuned = tune_reg_model(gbr, optimize='RMSE')

print('REG - Creating br Boost')
br = create_reg_model('br')
print('REG - Tuning br Boost')
br_tuned = tune_reg_model(br, optimize='RMSE')

print('REG - Creating ridge Boost')
ridge = create_reg_model('ridge')
print('REG - Tuning ridge Boost')
ridge_tuned = tune_reg_model(ridge, optimize='RMSE')


try:
    best_model_reg = compare_reg_models(#include=['gbr']
                                        include=['gbr', 'br', 'ridge'
                                                ,gbr_tuned, br_tuned, ridge_tuned]
                                                ,sort="RMSE")
except Exception as e:
    print(f"Error during model comparison: {e}")

plot_reg_model(best_model_reg, plot='feature')  # Feature Importance
plot_reg_model(best_model_reg, plot='error')   # Error Analysis
plot_reg_model(best_model_reg, plot='residuals')   # Residual Analysis
#plot_reg_model(best_model_reg, plot='feature_all')  # Feature Importance

# Prediction 
reg_prediction = predict_reg_model(best_model_reg, data=X_test)


# 3: Correcting value to within range of 0 to 365 
reg_prediction['prediction_label'] = np.round(reg_prediction['prediction_label']).astype(int)
reg_prediction['prediction_label'] = np.clip(reg_prediction['prediction_label'], 0, 365)

solution = pd.concat([test_df[['patient_id']], reg_prediction.iloc[:,-1]], axis=1)

solution_df = solution_df.merge(solution, left_on='patient_id', right_on='patient_id').drop(columns=['metastatic_diagnosis_period']).rename(columns={"prediction_label": "metastatic_diagnosis_period"})
solution_df['metastatic_diagnosis_period'] = solution_df['metastatic_diagnosis_period'].apply(lambda x: math.ceil(x))
solution_df.to_csv("./data/predictions.csv", index=False)


#### If you want to try ####
#### Try Neural Networks ####  

# import tensorflow as tf
# from tensorflow import keras
# from keras import layers
# from sklearn.model_selection import train_test_split
# from keras.callbacks import EarlyStopping

# _X_train, _X_test, _y_train, _y_test = train_test_split(X_train, y, test_size=0.2, random_state=42)

# input_shape = _X_train.shape[1]

# model = keras.Sequential([
#     layers.Input(shape=(input_shape,)),        # Input layer 
#     layers.Dense(32, activation='relu'),    
#     layers.Dense(1, activation='linear')       # Output layer with a single neuron (for regression)
# ])

# model.compile(optimizer='adam', loss='mean_squared_error', 
#               metrics=[keras.metrics.RootMeanSquaredError()])

# # early stopping callback
# es = EarlyStopping(monitor='val_loss',
#                    mode='min',
#                    patience=50,
#                    restore_best_weights = True)

# history = model.fit(_X_train, _y_train, epochs=6, batch_size=32, validation_data=(_X_test, _y_test))

# import matplotlib.pyplot as plt
# # let's see the training and validation accuracy by epoch
# history_dict = history.history
# loss_values = history_dict['loss'] # you can change this
# val_loss_values = history_dict['val_loss'] # you can also change this
# epochs = range(1, len(loss_values) + 1) # range of X (no. of epochs)
# plt.plot(epochs, loss_values, 'bo', label='Training loss')
# plt.plot(epochs, val_loss_values, 'orange', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# # scatterplot of actual vs. pred
# # specify the dimensions 
# fig, axes = plt.subplots(1,2) # 1 row, 2 columns

# # this makes the individual subplots
# # Training Results
# axes[0].scatter(x=_y_train, y=model.predict(_X_train)) #first row, first entry (left top)
# axes[0].set_xlabel("Actual", fontsize=10)
# axes[0].set_ylabel("Predicted",  fontsize=10)
# axes[0].set_title("Training")
# # add 45 deg line
# x = np.linspace(*axes[0].get_xlim())
# axes[0].plot(x, x, color='red')
# # Validation Results
# axes[1].scatter(x=_y_test, y=model.predict(_X_test)) # first row, second entry (right top)
# axes[1].set_xlabel("Actual", fontsize=10)
# axes[1].set_ylabel("Predicted",  fontsize=10)
# axes[1].set_title("Validation")
# # add 45 deg line
# x = np.linspace(*axes[1].get_xlim())
# axes[1].plot(x, x, color='red')

# # tight layout
# fig.tight_layout()

# # show the plot
# plt.show()

# # metrics
# pred = model.predict(_X_test)

# trainpreds = model.predict(_X_train)

# from sklearn.metrics import mean_absolute_error
# print(mean_absolute_error(_y_train, trainpreds)) # train
# print(mean_absolute_error(_y_test, pred)) # test


# # # Evaluate the model on the test data
# # test_loss = model.evaluate(X_test, y_test)
# # print(f"Test Loss: {test_loss:.4f}")

# NNpredictions = model.predict(X_test)
# print(NNpredictions.shape)

# solution_df['metastatic_diagnosis_period'] = pd.DataFrame(NNpredictions)
# solution_df['metastatic_diagnosis_period'] = np.round(solution_df['metastatic_diagnosis_period']).astype(int)
# solution_df['metastatic_diagnosis_period'] = np.clip(solution_df['metastatic_diagnosis_period'], 0, 365)
# solution_df.to_csv("./data/NNpredictions.csv", index=False)


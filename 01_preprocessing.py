import os
from backend import *

#####################
#Variables and Setup#
#####################
num_months = 12
data_folder = "data"
external_folder = "external"
train_file_name = "train.csv"
test_file_name = "test.csv"
data_json_file = "data.json"
zip_county_file_name = "zip_county.csv"
# Dictionary to store AQI files
aqi_county_files_name = {
    2013: "annual_aqi_by_county_2013.csv",
    2014: "annual_aqi_by_county_2014.csv",
    2015: "annual_aqi_by_county_2015.csv",
    2016: "annual_aqi_by_county_2016.csv",
    2017: "annual_aqi_by_county_2017.csv",
    2018: "annual_aqi_by_county_2018.csv"
}
disaster_file_name = "disaster.csv"
nri_file_name = "NRI_Table_Counties2020.csv"

train_file = os.path.join(data_folder, train_file_name)
test_file = os.path.join(data_folder, test_file_name)
zip_county_file = os.path.join(data_folder, external_folder, zip_county_file_name)
disaster_file = os.path.join(data_folder, external_folder, disaster_file_name)
nri_file = os.path.join(data_folder, external_folder, nri_file_name)

# Read JSON data
data = backend.read_json(data_json_file)

# Read data
train_df = backend.read_file(train_file, "csv")
test_df = backend.read_file(test_file, "csv")
zip_county_df = backend.read_file(zip_county_file, "csv", "ZIP", str)
# preprocess zip_county_df: zip -> zip3
zip_county_df = backend.zip_to_zip3(zip_county_df)
zip_county_df.to_csv("zip_county_df.csv")
print("zip_county_df is processed.")
disaster_df = backend.read_file(disaster_file, "csv")
nri_df = backend.read_file(nri_file, "csv")
update_nri_df = backend.get_nri_df(nri_df, data)
# Read AQI data for each year
aqi_county_dfs = {}
for year, filename in aqi_county_files_name.items():
    file_path = os.path.join(data_folder, external_folder, filename)
    aqi_county_dfs[year] = backend.read_file(file_path, "csv")

print("completed reading data")
# Merge nri and zip3 data
nri_zip3_df = update_nri_df.merge(zip_county_df, left_on="COUNTY_STATE", right_on="COUNTY_STATE")
nri_zip3_df.drop(columns=["COUNTYNAME", "STATE", "COUNTY_STATE"], inplace=True)
# Get mean NRI score by zip3
nri_score_grouped = nri_zip3_df.groupby("ZIP3")["RISK_SCORE"].mean()
nri_score_df = nri_score_grouped.reset_index()
# Manipulation
# Remove unnecessary columns
train_df.columns = data["update_columns_train"]
test_df.columns = data["update_columns_test"]
train_df.rename(columns={"patient_state": "state"}, inplace=True)
test_df.rename(columns={"patient_state": "state"}, inplace=True)
#################
# Missing values#
#################
# Fill missing values
# payer_type: create a new category - unknown
# patient_race: create a new category - unknown
train_df = backend.fill_missing_values_category(train_df, "payer_type", "unknown")
train_df = backend.fill_missing_values_category(train_df, "patient_race", "unknown")

test_df = backend.fill_missing_values_category(test_df, "payer_type", "unknown")
test_df = backend.fill_missing_values_category(test_df, "patient_race", "unknown")

# Average temperature: remove missing values
train_df.dropna(subset=data["tmp_columns"], inplace=True)
train_df.reset_index(inplace=True, drop=True)

#################
# Manipulation  #
#################
# Get extreme temperature days
print("Manipulating temperatures...")
train_df = backend.get_extremem_temp(train_df, data)
test_df = backend.get_extremem_temp(test_df, data)
print(train_df.columns.to_list())
train_df = backend.temp_agg(train_df, data)
test_df = backend.temp_agg(test_df, data)
# test_df.dropna(subset=data["tmp_columns"], inplace=True)
# test_df.reset_index(inplace=True, drop=True)
train_df["patient_zip3"] = train_df["patient_zip3"].astype(str)
test_df["patient_zip3"] = test_df["patient_zip3"].astype(str)
print(train_df.shape)
print(test_df.shape)
# replace income_individual_median missing values with its mean value
for aCol in data["missing_values_cols"]:
    print("Replacing missing values of ", aCol, "...")
    train_df[aCol] = train_df[aCol].fillna(train_df[aCol].mean())
    test_df[aCol] = test_df[aCol].fillna(test_df[aCol].mean())

# Calculate average temperature by month
test_df = backend.calculate_monthly_avg_temperature(test_df, data, num_months)
train_df = backend.calculate_monthly_avg_temperature(train_df, data, num_months)
# Calculate average temperature by year
test_df = backend.calculate_yearly_avg_temperature(test_df, data, num_months)
train_df = backend.calculate_yearly_avg_temperature(train_df, data, num_months)


# Remove unnecessary columns
train_df.drop(columns=data["tmp_columns"], inplace=True)
train_df.drop(columns=data["unnecessary_cols"], inplace=True)
test_df.drop(columns=data["tmp_columns"], inplace=True)
test_df.drop(columns=data["unnecessary_cols"], inplace=True)
print("Unnecessary columns are removed")

# preprocess disaster_df
disaster_df = backend.get_disaster_df(disaster_df, data)
# print(disaster_df)
print("disaster_df is processed.")

# Combine household income
print("Combining household income...")
train_df["income_35_below"] = 0
train_df["income_35_75"] = 0
train_df["income_75_above"] = 0
test_df["income_35_below"] = 0
test_df["income_35_75"] = 0
test_df["income_75_above"] = 0
for anIncomeCol in data["income_35_below"]:
    train_df["income_35_below"] += train_df[anIncomeCol]
    test_df["income_35_below"] += test_df[anIncomeCol]
for anIncomeCol in data["income_35_75"]:
    train_df["income_35_75"] += train_df[anIncomeCol]
    test_df["income_35_75"] += test_df[anIncomeCol]
for anIncomeCol in data["income_75_above"]:
    train_df["income_75_above"] += train_df[anIncomeCol]
    test_df["income_75_above"] += train_df[anIncomeCol]

# Remove unnecessary columns
train_df = train_df.drop(columns=data["columns_to_drop"])
# Drop columns from test_df
test_df = test_df.drop(columns=data["columns_to_drop"])

print(test_df.shape)
# merge aqi_county & zip_county & test_df
print("Merging AQI data...")
final_train_df = backend.merge_dataframes(aqi_county_dfs, data["state_abbreviations"], train_df, data)
final_test_df = backend.merge_dataframes(aqi_county_dfs, data["state_abbreviations"], test_df, data)
print(final_test_df.shape)
# merge final_train_df / final_test_df with disaster
print("Merging final_train_df...")
final_train_df = pd.merge(final_train_df, disaster_df, on='state', how='left')
final_test_df = pd.merge(final_test_df, disaster_df, on='state', how='left')
# convert string columns to lowercase
final_train_df = final_train_df.apply(lambda x: x.str.lower() if(x.dtype=="object") else x)
final_test_df = final_test_df.apply(lambda x: x.str.lower() if(x.dtype=="object") else x)
# add obesity feature, 0 = data missing, 1 = not obese, 2 = obese
final_train_df = backend.add_obesity_column(final_train_df, 30)
final_test_df = backend.add_obesity_column(final_test_df, 30)
# Merge NRI data to train and test
print("Merging NRI data...")
final_train_df = final_train_df.merge(nri_score_df, left_on="patient_zip3", right_on="ZIP3", how="left")
final_test_df = final_test_df.merge(nri_score_df, left_on="patient_zip3", right_on="ZIP3", how="left")
final_train_df.drop(columns=["ZIP3"], inplace=True)
final_test_df.drop(columns=["ZIP3"], inplace=True)
print(final_test_df.shape)
print("Final training and testing data are created.")

# Determine if the breast cancer diagnosis code is ICD 9 or 10
final_train_df["breast_cancer_diagnosis_code_isIcd"] = False
final_test_df["breast_cancer_diagnosis_code_isIcd"] = False
final_train_df["breast_cancer_diagnosis_code_isIcd"] = final_train_df["breast_cancer_diagnosis_code"].apply(backend.starts_with_letter)
final_test_df["breast_cancer_diagnosis_code_isIcd"] = final_test_df["breast_cancer_diagnosis_code"].apply(backend.starts_with_letter)
final_train_df["temp_change"] = final_train_df["temp_18"] - final_train_df["temp_13"]
final_train_df["yearly_temp_increase"] = (final_train_df["temp_18"] - final_train_df["temp_13"]) / 5
final_test_df["temp_change"] = final_test_df["temp_18"] - final_test_df["temp_13"]
final_test_df["yearly_temp_increase"] = (final_test_df["temp_18"] - final_test_df["temp_13"]) / 5
# Convert icd 9 to icd 10
print("Converting icd 9 to icd 10...")
final_train_df = backend.get_icd10(final_train_df)
final_test_df = backend.get_icd10(final_test_df)


# export the dataframe to csv
final_train_df.to_csv("./data/final_train_df.csv", index=False)
final_test_df.to_csv('./data/final_test_df.csv', index=False)
print("Shape of final_test_df: ", final_test_df.shape)
print("Final training and testing data are exported")

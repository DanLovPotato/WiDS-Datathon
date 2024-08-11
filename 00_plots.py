# importing packages
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

data_folder = "data"
train_file_name = "final_train_df.csv"
test_file_name = "final_test_df.csv"
preprocessed_train_name = "preprocessed_train_df.csv"
normalized_train_name = "normalized_train_df.csv"

train_file = os.path.join(data_folder, train_file_name)
test_file = os.path.join(data_folder, test_file_name)
preprocessed_train_file = os.path.join(data_folder, preprocessed_train_name)
normalized_train_file = os.path.join(data_folder, normalized_train_name)

# Load your datasets
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
preprocessed_train_df = pd.read_csv(preprocessed_train_file)
normalized_train_df = pd.read_csv(normalized_train_file)

def get_col(X):
    
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
    print(categorical_columns)
    numeric_columns = X.select_dtypes(exclude=['object', 'category']).columns.tolist()

    return numeric_columns
 
numeric_col = get_col(train_df)

# Setting up the visualization environment again
sns.set(style="whitegrid")

for col in numeric_col:

    # Visualization 1: Distribution of patient ages
    plt.figure(figsize=(12, 6))
    sns.histplot(train_df[col], bins=30, kde=True, color='skyblue')
    plt.title('Distribution of '+col)
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

numeric_col_p = get_col(preprocessed_train_df)

for col in numeric_col_p:
    # Visualization 1: Distribution of patient ages
    plt.figure(figsize=(12, 6))
    sns.histplot(preprocessed_train_df[col], bins=30, kde=True, color='green')
    plt.title('Distribution of '+col)
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

dummy_items = ['no_farmer', 'breast_cancer_diagnosis_code_isIcd'
                               ,'aqi_median_is_outlier','aqi_90_pctl_is_outlier','aqi_max_is_outlier'
                               ,'has_extreme_hot_days']

for i in dummy_items:
    numeric_col_p.remove(i)

for col in numeric_col_p:
    # Visualization 1: Distribution of patient ages
    plt.figure(figsize=(12, 6))
    sns.histplot(normalized_train_df[col], bins=30, kde=True, color='red')
    plt.title('Distribution of '+col)
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    
print(normalized_train_df.columns.tolist())

import pandas as pd
import numpy as np
import os
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
# from statsmodels.stats.outliers_influence import variance_inflation_factor 


# ow on processed file

data_folder = "data"
file_name = "normalized_train_df.csv"

file = os.path.join(data_folder, file_name)
df = pd.read_csv(file)
pca = PCA().fit(df)
principalComponents_df = pca.fit(df)
explained_variance_ratio = pca.explained_variance_ratio_
total_explained_variance_ratio = explained_variance_ratio.sum()

cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1

# Print results
print(f"\nExplained Variance Ratio:\n{explained_variance_ratio}")
print(f"Total Explained Variance Ratio: {total_explained_variance_ratio:.4f}")
print(d)

# Plot explained variance ratio
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
plt.plot(cumulative_variance_ratio, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio by Principal Components')
plt.show()


pca_final = PCA(n_components=44)
principalComponents_df = pca_final.fit_transform(df)
principal_breast_Df = pd.DataFrame(data = principalComponents_df
             , columns = ['pc1', 'pc2','pc3', 'pc4','pc5', 'pc6','pc7'])

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pc1", y="pc2",
    data=principal_breast_Df,
    legend="full",
    alpha=0.3
)

# plt.legend(targets,prop={'size': 15})



# # VIF dataframe 
# vif_data = pd.DataFrame() 
# vif_data["feature"] = df.columns 
  
# # calculating VIF for each feature 
# vif_data["VIF"] = [variance_inflation_factor(df.values, i) 
#                           for i in range(len(df.columns))] 

# print(vif_data)


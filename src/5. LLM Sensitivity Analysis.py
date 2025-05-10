# %%
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


# %%
#Import cleaned dataset
df_analysis = pd.read_excel("../data/dataset_processed/export_cleaned.xlsx")
# %%
def compute_similarity_matrix(df, columns):
    # Initialize an empty DataFrame for the similarity matrix
    similarity_matrix = pd.DataFrame(0, index=columns, columns=columns)
    
    # Compute pairwise similarity
    for col1 in columns:
        for col2 in columns:
            similarity_matrix.loc[col1, col2] = (df[col1] == df[col2]).sum()
    
    return similarity_matrix

# Define three lists of columns to compare
columns_list_1 = [
    "problem_classification_phi3_cleaned",
    "problem_classification_llama3.1_cleaned",
    "problem_classification_mistral_cleaned",
    "problem_classification_qwen2_cleaned",
    "problem_classification_deepseek-r1:7b_cleaned"

]

columns_list_2 = [
    "method_classification_phi3_cleaned",
    "method_classification_llama3.1_cleaned",
    "method_classification_mistral_cleaned",
    "method_classification_qwen2_cleaned",
    "method_classification_deepseek-r1:7b_cleaned"

]

columns_list_3 = [
    "method_classification_phi3_AnalyticsFamily",
    "method_classification_llama3.1_AnalyticsFamily",
    "method_classification_mistral_AnalyticsFamily",
    "method_classification_qwen2_AnalyticsFamily",
    "method_classification_deepseek-r1:7b_AnalyticsFamily"

]

# Compute similarity matrices for each list

similarity_matrix_1 = compute_similarity_matrix(df_analysis, columns_list_1)
similarity_matrix_1.columns = [col.replace("_cleaned", "").split("_")[-1] for col in similarity_matrix_1.columns]
similarity_matrix_1.index = [col.replace("_cleaned", "").split("_")[-1] for col in similarity_matrix_1.index]
    


similarity_matrix_2 = compute_similarity_matrix(df_analysis, columns_list_2)
similarity_matrix_2.columns = [col.replace("_cleaned", "").split("_")[-1] for col in similarity_matrix_2.columns]
similarity_matrix_2.index = [col.replace("_cleaned", "").split("_")[-1] for col in similarity_matrix_2.index]
    

similarity_matrix_3 = compute_similarity_matrix(df_analysis, columns_list_3)
similarity_matrix_3.columns = [col.replace("_AnalyticsFamily", "").split("_")[-1] for col in similarity_matrix_3.columns]
similarity_matrix_3.index = [col.replace("_AnalyticsFamily", "").split("_")[-1] for col in similarity_matrix_3.index]
    
# Define custom colormap
colors = [(1, 1, 1), (0.7, 0.5, 0.85), (0.4, 0.2, 0.6), (0.5, 0.3, 0.1), (0.9, 0.6, 0)]
cmap = LinearSegmentedColormap.from_list("custom_white_puor", colors, N=100)

# Function to mask the upper triangle of a matrix
def mask_upper_triangle(matrix):
    mask = pd.DataFrame(False, index=matrix.index, columns=matrix.columns)
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            mask.iloc[i, j] = True
    return mask

# Plot the heatmaps in a single row
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

sns.heatmap(similarity_matrix_1, annot=True, fmt="d", cmap=cmap, mask=mask_upper_triangle(similarity_matrix_1), ax=axes[0])
axes[0].set_title("Sensitivity Analysis - Problem Classification")
axes[0].tick_params(axis='y', rotation=0)

sns.heatmap(similarity_matrix_2, annot=True, fmt="d", cmap=cmap, mask=mask_upper_triangle(similarity_matrix_2), ax=axes[1])
axes[1].set_title("Sensitivity Analysis - Method Classification")
axes[1].tick_params(axis='y', rotation=0)

sns.heatmap(similarity_matrix_3, annot=True, fmt="d", cmap=cmap, mask=mask_upper_triangle(similarity_matrix_3), ax=axes[2])
axes[2].set_title("Sensitivity Analysis - SCS Classification")
axes[2].tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.show()

# %%

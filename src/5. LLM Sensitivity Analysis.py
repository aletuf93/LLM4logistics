# %%
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

from itertools import combinations
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import LabelEncoder

from scipy.stats import chi2_contingency

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

# %% Jaccard similarity
def compute_jaccard_similarity_matrix(df: pd.DataFrame, columns_list: list) -> pd.DataFrame:
    """
    Calcola la similarità di Jaccard tra tutte le coppie di classificatori elencati in columns_list.

    Args:
        df (pd.DataFrame): DataFrame che contiene le colonne delle predizioni.
        columns_list (list): Lista dei nomi delle colonne dei classificatori.

    Returns:
        pd.DataFrame: Matrice di similarità Jaccard (simmetrica) tra ogni coppia di classificatori.
    """
    # Creazione della matrice vuota
    jaccard_matrix = pd.DataFrame(index=columns_list, columns=columns_list, dtype=float)

    # Itera su tutte le coppie possibili di classificatori
    for col1, col2 in combinations(columns_list, 2):
        # Calcola il Jaccard index sulle etichette categoriali
        score = jaccard_score(
            df[col1], df[col2], average='macro'  # macro: media tra classi
        )
        jaccard_matrix.loc[col1, col2] = score
        jaccard_matrix.loc[col2, col1] = score  # simmetrico

    # Diagonale = 1.0 (stessa colonna)
    for col in columns_list:
        jaccard_matrix.loc[col, col] = 1.0

    return jaccard_matrix


def encode_labels(df: pd.DataFrame, columns_list: list) -> pd.DataFrame:
    df_encoded = df.copy()
    for col in columns_list:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
    return df_encoded

# %% Calculate Jaccard matrix for problem classification
df_encoded = encode_labels(df_analysis, columns_list_1)
jaccard_sim_matrix = compute_jaccard_similarity_matrix(df_encoded, columns_list_1)

print(jaccard_sim_matrix.round(3))
# %% Calculate Jaccard matrix for method classification

df_encoded = encode_labels(df_analysis, columns_list_2)
jaccard_sim_matrix = compute_jaccard_similarity_matrix(df_encoded, columns_list_2)

print(jaccard_sim_matrix.round(3))

# %% Calculate Jaccard matrix for analytics family

df_encoded = encode_labels(df_analysis, columns_list_3)
jaccard_sim_matrix = compute_jaccard_similarity_matrix(df_encoded, columns_list_3)

print(jaccard_sim_matrix.round(3))


# %% Chi2 function

def chi2_test_pairs(df, columns):
    results = []
    for col1, col2 in combinations(columns, 2):
        # Costruisci la tabella di contingenza
        contingency_table = pd.crosstab(df[col1], df[col2])
        
        # Esegui il test del chi quadrato
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        results.append({
            "Model_1": col1,
            "Model_2": col2,
            "Chi2_stat": chi2,
            "p_value": p_value,
            "dof": dof
        })
    return pd.DataFrame(results)


# Calculate Chi2 test for problem classification
chi2_results = chi2_test_pairs(df_analysis, columns_list_1)
print(chi2_results)
# %%
chi2_results = chi2_test_pairs(df_analysis, columns_list_2)
print(chi2_results)

# %%
chi2_results = chi2_test_pairs(df_analysis, columns_list_3)
print(chi2_results)

# %%

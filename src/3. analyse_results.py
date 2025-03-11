# %%
import pandas as pd
import itertools

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# %%
#Import cleaned dataset
df_cleaned = pd.read_excel("../data/dataset_processed/export_cleaned.xlsx")

# %% define function for heatmaps

def createHeatmap(D_scs: pd.DataFrame,
                  title: str,
                  column_problem: str,
                  column_method: str,
                  listProblems: list,
                  listMethods: list,
                  ax):  # Nuovo parametro ax
    
    # Generazione delle combinazioni possibili
    permutations = [(p, m) for p in listProblems for m in listMethods]
    D = pd.DataFrame(permutations, columns=[column_problem, column_method])
    D['indice'] = [f"{D.iloc[i][column_problem]}_{D.iloc[i][column_method]}" for i in range(len(D))]

    # Filtraggio dei dati
    D_scs = D_scs[D_scs[column_problem].isin(listProblems)]
    D_scs = D_scs[D_scs[column_method].isin(listMethods)]

    # Selezione delle colonne utili
    D_scs = D_scs[[column_problem, column_method]]

    # Raggruppamento e conteggio
    D_scs = D_scs.groupby([column_problem, column_method]).size().reset_index(name='count')
    D_scs['indice'] = [f"{D_scs.iloc[i][column_problem]}_{D_scs.iloc[i][column_method]}" for i in range(len(D_scs))]

    # Aggiunta delle combinazioni mancanti
    for indice in set(D['indice']):
        if indice not in set(D_scs['indice']):
            method = D[D['indice'] == indice][column_method].iloc[0]
            pattern = D[D['indice'] == indice][column_problem].iloc[0]
            D_scs = pd.concat([D_scs, pd.DataFrame([[pattern, method, 0, indice]], columns=D_scs.columns)])

    # Creazione della matrice pivot
    D_scs_square = D_scs.pivot(index=column_problem, columns=column_method, values='count')

    # Ordinamento delle righe
    D_scs_square['indiceSorting'] = [D_scs_square.index[i][1:] for i in range(len(D_scs_square))]
    D_scs_square['indiceSorting'] = D_scs_square['indiceSorting'].astype(str)
    D_scs_square.sort_values(by='indiceSorting', inplace=True)
    D_scs_square.drop(columns=['indiceSorting'], inplace=True)

    # Creazione della mappa di colori personalizzata
    colors = [(1, 1, 1), (0.7, 0.5, 0.85), (0.4, 0.2, 0.6), (0.5, 0.3, 0.1), (0.9, 0.6, 0)]
    cmap = LinearSegmentedColormap.from_list("custom_white_puor", colors, N=100)

    # Plot della heatmap sul grafico fornito
    sns.heatmap(D_scs_square,
                linewidths=.5,
                annot=True,
                fmt="d",
                cmap=cmap,
                mask=(D_scs_square == 0),
                vmin=0,
                vmax=50,
                ax=ax)  # Passiamo ax invece di plt.figure()

    # Impostazione delle etichette
    ax.set_xlabel("Method")
    ax.set_ylabel("Problem")
    ax.set_title(title)

# %% create plot heatmaps
columns_analysis = ['Year',
                    'problem_classification_phi3_cleaned',
                    'problem_classification_llama3.1_cleaned',
                    'problem_classification_mistral_cleaned',
                    'problem_classification_qwen2_cleaned',
                    'problem_classification_deepseek-r1:7b_cleaned',
                    'method_classification_phi3_cleaned',
                    'method_classification_llama3.1_cleaned',
                    'method_classification_mistral_cleaned',
                    'method_classification_qwen2_cleaned',
                    'method_classification_deepseek-r1:7b_cleaned',
                    'Supply chain System']
df_analysis = df_cleaned[columns_analysis]


# Define all permutations for matrices

listProblems= ["P1", "P2", "P3", "P4", "P5", "P6"]
listMethods = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "CT1", "OR1", "OR2"]
permutations = list(itertools.product(listProblems, listMethods))


dict_column_models = {"phi3": {"methods_columns":"method_classification_phi3_cleaned",
                              "problems_columns":"problem_classification_phi3_cleaned"},
                      "llama3.1": {"methods_columns": "method_classification_llama3.1_cleaned",
                                   "problems_columns": "problem_classification_llama3.1_cleaned"},
                      "mistral": {"methods_columns": "method_classification_mistral_cleaned",
                                  "problems_columns": "problem_classification_mistral_cleaned"}
                      ,"qwen2": {"methods_columns":"method_classification_qwen2_cleaned",
                                "problems_columns": "problem_classification_qwen2_cleaned"},
                      "deepseek-r1": {"methods_columns":"method_classification_deepseek-r1:7b_cleaned",
                                "problems_columns": "problem_classification_deepseek-r1:7b_cleaned"}
}
import matplotlib.pyplot as plt

scs = ["PRODUCTION", "NETWORK", "WAREHOUSE"]
numLLMs = len(scs) + 1  # Una riga in più per la heatmap "overall"
numModels = len(dict_column_models)

# Creazione della figura con subplots
fig, axs = plt.subplots(numLLMs, numModels, figsize=(20, 16))

# Se la matrice axs ha una sola colonna, trasformiamola in un array 2D per evitare errori
if numModels == 1:
    axs = axs[:, None]

for i_column, model in enumerate(dict_column_models.keys()):

    # Plot heatmap con filtro per scs
    for i_row, supply_chain_system in enumerate(scs):
        df_filtered = df_analysis[df_analysis["Supply chain System"] == supply_chain_system]
        title = f"{supply_chain_system}_{model}"

        ax = axs[i_row, i_column]  # Passiamo l'asse corretto
        createHeatmap(df_filtered,
                      title=title,
                      column_problem=dict_column_models[model]["problems_columns"],
                      column_method=dict_column_models[model]["methods_columns"],
                      listProblems=listProblems,
                      listMethods=listMethods,
                      ax=ax)  # Passiamo ax

        # Settiamo le etichette
        if i_column == 0:
            ax.set_ylabel("Problem")

    # Plot heatmap senza filtro per scs
    title = f"overall_{model}"
    ax = axs[numLLMs - 1, i_column]  # Ultima riga

    createHeatmap(df_analysis,
                  title=title,
                  column_problem=dict_column_models[model]["problems_columns"],
                  column_method=dict_column_models[model]["methods_columns"],
                  listProblems=listProblems,
                  listMethods=listMethods,
                  ax=ax)  # Passiamo ax

    # Etichette finali
    ax.set_xlabel("Method")
    if i_column == 0:
        ax.set_ylabel("Problem")

# Miglioriamo il layout e mostriamo la figura
plt.tight_layout()
plt.savefig("../data/output/_master_heatmap.jpg")
plt.show()


# %% create timeline

df_analysis ['Decade'] = df_analysis['Year']//10*10

# correct the column "method" to use the new overall vote column
def add_unique_columns(df, *columns):
    """
    Modifica la funzione per accettare un numero variabile di colonne.
    Crea nuove colonne per ogni valore unico presente nelle colonne selezionate.
    
    :param df: DataFrame di input
    :param columns: Nomi delle colonne da analizzare
    :return: DataFrame con nuove colonne per ciascun valore unico
    """
    if not columns:
        raise ValueError("Devi specificare almeno una colonna.")

    # Seleziona le colonne specificate
    colonne_selezionate = df[list(columns)]

    # Trova tutti i valori unici nelle colonne selezionate
    valori_unici = pd.unique(colonne_selezionate.values.ravel())

    # Crea una nuova colonna per ogni valore unico
    for valore in valori_unici:
        df[valore] = colonne_selezionate.apply(lambda row: (row == valore).sum(), axis=1)

    return df


def max_score_columns_tiebreak_custom(df, columns_to_consider, new_column_name):
    """
    Function to calculate, for each row of the dataframe, the columns with the maximum score
    among those specified in 'columns_to_consider'. Returns a new column with the names
    of the columns that have the maximum score (list in case of a tie).
    
    :param df: Input DataFrame
    :param columns_to_consider: List of columns to consider for max score calculation
    :param new_column_name: The name of the new column where the max score result will be stored
    :return: DataFrame with the additional column specified by 'new_column_name'
    """

    # Select only the columns specified by the user
    selected_columns = df[columns_to_consider]

    # Create a new column with the specified name containing the columns with the max score for each row
    df[new_column_name] = selected_columns.apply(lambda row: selected_columns.columns[row == row.max()].tolist(), axis=1)

    return df

df_analysis = add_unique_columns(df_analysis, 
                                 'problem_classification_qwen2_cleaned',
                                 'problem_classification_deepseek-r1:7b_cleaned'
                                 
                                                      )

df_analysis = max_score_columns_tiebreak_custom(df=df_analysis,
                                                columns_to_consider=listProblems,
                                                new_column_name='problem_overall')

df_analysis = add_unique_columns(df_analysis,
                                 'method_classification_qwen2_cleaned',
                                 'method_classification_deepseek-r1:7b_cleaned'
                                              )

df_analysis = max_score_columns_tiebreak_custom(df=df_analysis,
                                                columns_to_consider=listMethods,
                                                new_column_name='method_overall')
# %% Generate all permutations

def generate_permutations_for_columns(df, method_col, problem_col, decade_col):
    """
    Generates all possible permutations between the lists in the method and problem columns
    while duplicating the corresponding decade values.
    
    :param df: Input DataFrame containing 'method_overall', 'problem_overall', and 'Decade' columns
    :param method_col: The name of the column containing lists of methods
    :param problem_col: The name of the column containing lists of problems
    :param decade_col: The name of the column containing the decade values
    :return: A new DataFrame with all possible permutations between method and problem columns
    """

    # Create a list to store the new rows
    new_rows = []

    # Iterate through each row in the dataframe
    for index, row in df.iterrows():
        # Get the lists from method and problem columns
        method_list = row[method_col]
        problem_list = row[problem_col]
        decade_value = row[decade_col]

        # Generate all possible combinations between the methods and problems
        for method, problem in itertools.product(method_list, problem_list):
            # Create a new row for each combination, duplicating the decade
            new_row = {
                decade_col: decade_value,
                method_col: method,
                problem_col: problem
            }
            new_rows.append(new_row)

    # Convert the list of new rows into a DataFrame
    result_df = pd.DataFrame(new_rows)

    return result_df

df_permutations = df_analysis[["Decade", "method_overall", "problem_overall"]]
df_analysis_permutations = generate_permutations_for_columns(df=df_permutations,
                                                             method_col='method_overall',
                                                             problem_col='problem_overall',
                                                             decade_col='Decade')
# %% plot overall heatmap
# Creazione di una nuova figura e asse
fig, ax = plt.subplots(figsize=(5, 3))  # Regola la dimensione se necessario

# Creazione della heatmap indipendente
createHeatmap(df_analysis_permutations,
              title="Overall classification",
              column_problem='problem_overall',
              column_method='method_overall',
              listProblems=listProblems,
              listMethods=listMethods,
              ax=ax)  # Passiamo esplicitamente l'asse

# Salvataggio del grafico
plt.savefig("../data/output/_overall_heatmap.jpg")
plt.show()  # Mostra il grafico
 

# %%
D_years = df_analysis_permutations.groupby(['Decade','method_overall']).size().reset_index()
D_years_square = D_years.pivot('method_overall',columns='Decade',values = 0)
#D_years_square.drop(columns=[2030],inplace=True)

#plot heatmap
plt.figure()
sns.heatmap(D_years_square, linewidths=.5,annot=True,cmap="YlOrRd", fmt='g')
plt.title("Transition of the methods implementation over the time")
plt.savefig(f"../data/output/_time_transition.jpg") 

# %%

# %% Create graphs Analytics Families
# Dizionario aggiornato per il nuovo mapping
analytics_family_columns = {
    "phi3": {
        "methods_columns": "method_classification_phi3_cleaned",
        "analytics_columns": "method_classification_phi3_AnalyticsFamily"
    },
    "llama3.1": {
        "methods_columns": "method_classification_llama3.1_cleaned",
        "analytics_columns": "method_classification_llama3.1_AnalyticsFamily"
    },
    "mistral": {
        "methods_columns": "method_classification_mistral_cleaned",
        "analytics_columns": "method_classification_mistral_AnalyticsFamily"
    },
    "qwen2": {
        "methods_columns": "method_classification_qwen2_cleaned",
        "analytics_columns": "method_classification_qwen2_AnalyticsFamily"
    },
    "deepseek-r1": {
        "methods_columns": "method_classification_deepseek-r1:7b_cleaned",
        "analytics_columns": "method_classification_deepseek-r1:7b_AnalyticsFamily"
    }
}

# Lista delle famiglie di metodi
listAnalytics = ["DESCRIPTIVE", "PREDICTIVE", "PRESCRIPTIVE"]

# Creazione del nuovo DataFrame con solo le colonne utili
columns_analysis = [
    'Year',
    'method_classification_phi3_cleaned',
    'method_classification_llama3.1_cleaned',
    'method_classification_mistral_cleaned',
    'method_classification_qwen2_cleaned',
    'method_classification_deepseek-r1:7b_cleaned',
    'method_classification_phi3_AnalyticsFamily',
    'method_classification_llama3.1_AnalyticsFamily',
    'method_classification_mistral_AnalyticsFamily',
    'method_classification_qwen2_AnalyticsFamily',
    'method_classification_deepseek-r1:7b_AnalyticsFamily',
    'Supply chain System'
]
df_analysis2 = df_cleaned[columns_analysis]

# Creazione dei nuovi heatmap
fig, axs = plt.subplots(numLLMs, len(analytics_family_columns), figsize=(20, 16))
if len(analytics_family_columns) == 1:
    axs = axs[:, None]  # Evita errori con un solo modello

for i_column, model in enumerate(analytics_family_columns.keys()):
    for i_row, supply_chain_system in enumerate(scs):
        df_filtered = df_analysis2[df_analysis2["Supply chain System"] == supply_chain_system]
        title = f"{supply_chain_system}_{model}_AnalyticsFamily"
        ax = axs[i_row, i_column]
        createHeatmap(df_filtered,
                      title=title,
                      column_problem=analytics_family_columns[model]["methods_columns"],
                      column_method=analytics_family_columns[model]["analytics_columns"],
                      listProblems=listMethods,  # Qui listMethods rappresenta i metodi originali
                      listMethods=listAnalytics,
                      ax=ax)
        if i_column == 0:
            ax.set_ylabel("Method")
    
    # Heatmap complessiva senza filtro "Supply chain System"
    title = f"overall_{model}_AnalyticsFamily"
    ax = axs[numLLMs - 1, i_column]
    createHeatmap(df_analysis2,
                  title=title,
                  column_problem=analytics_family_columns[model]["methods_columns"],
                  column_method=analytics_family_columns[model]["analytics_columns"],
                  listProblems=listMethods,
                  listMethods=listAnalytics,
                  ax=ax)
    if i_column == 0:
        ax.set_ylabel("Method")
    ax.set_xlabel("Analytics Family")

plt.tight_layout()
plt.savefig("../data/output/_method_analytics_family_heatmap.jpg")
plt.show()

# %% Grafico con variazioni percentuali
# %% Calcolo variazioni percentuali tra decadi

# Raggruppa i dati e calcola i conteggi
D_years = df_analysis_permutations.groupby(['Decade', 'method_overall']).size().reset_index(name='count')

# Trasforma in una tabella pivot (decade come colonne)
D_years_square = D_years.pivot(index='method_overall', columns='Decade', values='count')

# Calcola la variazione percentuale da una decade all'altra (ignorando la prima)
D_years_square_pct = D_years_square.pct_change(axis=1) * 100  # Converti in percentuale

# Rimuove la prima colonna (senza riferimento precedente per la variazione)
D_years_square_pct = D_years_square_pct.iloc[:, 1:]

# Plot della heatmap con variazioni percentuali
plt.figure(figsize=(6, 4))
sns.heatmap(D_years_square_pct, linewidths=.5, annot=True, cmap="coolwarm", fmt=".1f", center=0)
plt.title("Percentual Variation of Methods Implementation Over Time")
plt.ylabel("Method")
plt.xlabel("Decade")
plt.savefig("../data/output/_time_transition_percentage.jpg")
plt.show()


# %%

df_analysis2 = add_unique_columns(df_analysis2, 
                                 'method_classification_qwen2_AnalyticsFamily',
                                 'method_classification_deepseek-r1:7b_AnalyticsFamily'
                                 
                                                      )

df_analysis2 = max_score_columns_tiebreak_custom(df=df_analysis2,
                                                columns_to_consider=listAnalytics,
                                                new_column_name='analytics_overall')

df_analysis2 = add_unique_columns(df_analysis2, 
                                 'method_classification_qwen2_cleaned',
                                 'method_classification_deepseek-r1:7b_cleaned'
                                 
                                                      )

df_analysis2 = max_score_columns_tiebreak_custom(df=df_analysis2,
                                                columns_to_consider=listMethods,
                                                new_column_name='method_overall')


# %%
df_analysis2 ['Decade'] = df_analysis2['Year']//10*10
df_permutations = df_analysis2[["Decade", "method_overall", "analytics_overall"]]
df_analysis_permutations = generate_permutations_for_columns(df=df_permutations,
                                                             method_col='method_overall',
                                                             problem_col='analytics_overall',
                                                             decade_col='Decade')
# %% plot overall heatmap
# Creazione di una nuova figura e asse
fig, ax = plt.subplots(figsize=(5, 3))  # Regola la dimensione se necessario

# Creazione della heatmap indipendente
createHeatmap(df_analysis_permutations,
              title="Overall classification",
              column_problem='analytics_overall',
              column_method='method_overall',
              listProblems=listAnalytics,
              listMethods=listMethods,
              ax=ax)  # Passiamo esplicitamente l'asse

# Salvataggio del grafico
plt.savefig("../data/output/_overall_analytics_heatmap.jpg")
plt.show()  # Mostra il grafico
 


# %%

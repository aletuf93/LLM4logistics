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
def createHeatmap(D_scs : pd.DataFrame,
                  title : str,
                  column_problem: str,
                  column_method: str,
                  listProblems: list,
                  listMethods: list):
    
    ## Generate permutations dataframe
    D = pd.DataFrame(permutations,columns = [column_problem, column_method])
    D['indice'] = [f"{D.iloc[i][column_problem]}_{D.iloc[i][column_method]}" for i in range(0,len(D))]

    ## Clean analysis dataframe from dirty data
    D_scs = D_scs[D_scs[column_problem].isin(listProblems)]
    D_scs = D_scs[D_scs[column_method].isin(listMethods)]

    ## Reduce to useful columns
    D_scs = D_scs[[column_problem, column_method]]

    ## group by tuple method-problem
    D_scs = D_scs.groupby([column_problem, column_method]).size().reset_index()
    D_scs['indice'] = [f"{D_scs.iloc[i][column_problem]}_{D_scs.iloc[i][column_method]}" for i in range(0,len(D_scs))]
    D_scs.rename(columns={0:'count'},inplace=True)
    
    ## match with index dataframe
    for indice in set(D['indice']):
        if indice not in set(D_scs['indice']):
            method = D[D['indice']==indice][column_method].iloc[0]
            pattern = D[D['indice']==indice][column_problem].iloc[0]
            D_scs = D_scs.append(pd.DataFrame([[pattern, method,0,indice]],columns=D_scs.columns))
    
    
    ## define a from-to square matrix
    D_scs_square = D_scs.pivot(column_problem, columns=column_method,values = 'count')
    
    #sort rows by P*
    D_scs_square['indiceSorting'] = [D_scs_square.index[i][1:] for i in range(0,len(D_scs_square))]
    D_scs_square['indiceSorting'] = D_scs_square['indiceSorting'].astype(float)
    D_scs_square.sort_values(by='indiceSorting',inplace=True)
    D_scs_square.drop(columns=['indiceSorting'],inplace=True)
    
    # define a customize colormap
    # # Create a custom colormap that starts with white and transitions to PuOr colors
    colors = [(1, 1, 1), (0.7, 0.5, 0.85), (0.4, 0.2, 0.6), (0.5, 0.3, 0.1), (0.9, 0.6, 0)]
    n_bins = 100  # Discretize the colormap
    cmap_name = 'custom_white_puor'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins) 

    #plot heatmap
    #plt.figure()
    heatmap = sns.heatmap(D_scs_square,
                          linewidths=.5,
                          annot=True,
                          fmt="d",
                          #cmap="PuOr",
                          cmap=cmap,
                          mask=(D_scs_square == 0),
                          vmin=0,
                          vmax=50)
    
    heatmap.set_title(title)
    heatmap.set_xlabel(" ")
    heatmap.set_ylabel(" ")

    #plt.savefig(f"../data/output/Heatmap_{title}.png")
    return heatmap


# %% create plot heatmaps
columns_analysis = ['Year',
                    'problem_classification_phi3_cleaned',
                    'problem_classification_llama3.1_cleaned',
                    'problem_classification_mistral_cleaned',
                    'problem_classification_qwen2_cleaned',
                    'method_classification_phi3_cleaned',
                    'method_classification_llama3.1_cleaned',
                    'method_classification_mistral_cleaned',
                    'method_classification_qwen2_cleaned',
                    'Supply chain System']
df_analysis = df_cleaned[columns_analysis]


# Define all permutations for matrices

listProblems= ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"]
listMethods = ["D1", "D2", "D3", "D4", "PD1", "PD2", "PD3", "PS1", "PS2", "PS3", "PS4"]
permutations = list(itertools.product(listProblems, listMethods))


dict_column_models = {"phi3": {"methods_columns":"method_classification_phi3_cleaned",
                              "problems_columns":"problem_classification_phi3_cleaned"},
                      "llama3.1": {"methods_columns": "method_classification_llama3.1_cleaned",
                                   "problems_columns": "problem_classification_llama3.1_cleaned"},
                      "mistral": {"methods_columns": "method_classification_mistral_cleaned",
                                  "problems_columns": "problem_classification_mistral_cleaned"},
                      "qwen2": {"methods_columns":"method_classification_qwen2_cleaned",
                                "problems_columns": "problem_classification_qwen2_cleaned"}
}
scs = ["PRODUCTION", "NETWORK", "WAREHOUSE"]

# create comprehensive figure
fig, axs = plt.subplots(4, 4, figsize=(20, 20))

for i_column, model in enumerate(dict_column_models.keys()):

    # Plot heatmap with filte for scs
    for i_row, supply_chain_system in enumerate(scs):
        df_filtered = df_analysis[df_analysis["Supply chain System"] == supply_chain_system]

        title = f"{supply_chain_system}_{model}"
        
        # set the current axis
        plt.sca(axs[i_row, i_column])

        createHeatmap(df_filtered,
                    title=title,
                    column_problem=dict_column_models[model]["problems_columns"],
                    column_method=dict_column_models[model]["methods_columns"],
                    listProblems=listProblems,
                    listMethods=listMethods)
        
        # set y-label
        if i_column==0:
            axs[i_row, i_column].set_ylabel("Problem")
        
        
    
    
    # Plot heatmap without filter for scs
    title = f"overall_{model}"

    # set the current axis
    plt.sca(axs[3, i_column])

    createHeatmap(df_analysis,
                title=title,
                column_problem=dict_column_models[model]["problems_columns"],
                column_method=dict_column_models[model]["methods_columns"],
                listProblems=listProblems,
                listMethods=listMethods)
    # set x-label
    axs[3, i_column].set_xlabel("Method")
    # set y-label
    if i_column==0:
        axs[3, i_column].set_ylabel("Problem")
plt.savefig(f"../data/output/_master_heatmap.jpg")

# %% Vote and aggregate 
# Seleziona le colonne di interesse
colonne = [
    'problem_classification_phi3_cleaned',
    'problem_classification_llama3.1_cleaned',
    'problem_classification_mistral_cleaned',
    'problem_classification_qwen2_cleaned'
]

# Unisci tutte le colonne in una "long format" dataframe
df_long = df_analysis.melt(id_vars=[], value_vars=colonne, var_name='source', value_name='classification')

    # Aggiungi un ID riga per mantenere la tracciabilità delle righe originali
    df_long['row_id'] = df.index

    # Crea la tabella pivot che conta le occorrenze per riga e classificazione
    df_pivot = df_long.pivot_table(index='row_id', columns='classification', aggfunc='size', fill_value=0)

    # Unisci i risultati della pivot con il dataframe originale, mantenendo l'ordine delle righe
    df_finale = df.join(df_pivot, on=df.index)

    return df_finale

# Chiamare la funzione con il percorso del file
file_path = '/path_to_your_file/export_cleaned.xlsx'
df_modificato = conta_valori_colonne_con_pivot(file_path)

# Visualizzare il dataframe con le nuove colonne
print(df_modificato.head())

# %% create timeline

df_analysis ['Decade'] = df_analysis['Year']//10*10
# correct the column "method" to use the new overall vote column
D_years = D_literature.groupby(['Decade','Method']).size().reset_index()
D_years_square = D_years.pivot('Method',columns='Decade',values = 0)
D_years_square.drop(columns=[2020],inplace=True)

#plot heatmap
plt.figure()
sns.heatmap(D_years_square, linewidths=.5,annot=True,cmap="YlOrRd")
plt.title("Transition of the methods implementation over the time")
plt.savefig(f"timeTransition.png")
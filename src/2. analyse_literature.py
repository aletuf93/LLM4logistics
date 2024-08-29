#%% Import packages and data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import itertools

# %% import data
df_processed = pd.read_excel("../data/dataset_processed/export_provvisorio_20240825.xlsx")

# %% clean data
columns_to_clean = ["problem_classification_phi3", "problem_classification_phi3"]

def clean_problem_classification(string):
    if "P10" in string:
        return "P10"
    elif "P1" in string:
        return "P1"
    elif "P2" in string:
        return "P2"
    elif "P3" in string:
        return "P3"
    elif "P4" in string:
        return "P4"
    elif "P5" in string:
        return "P5"
    elif "P6" in string:
        return "P6"
    elif "P7" in string:
        return "P7"
    elif "P8" in string:
        return "P8"
    elif "P9" in string:
        return "P9"
    elif "OTHER" in string:
        return "OTHER"
    else:
        return "CLASSIFICATION ERROR"
    
df_cleaned = df_processed

for column in columns_to_clean:
    df_cleaned[f"{column}_cleaned"] = [clean_problem_classification(i) for i in df_processed[column]]



#%% Define all permutations

listPatterns = list(set(D_literature['Decision Pattern']))
listMethods = list(set(D_literature['Method']))
permutations = list(itertools.product(listPatterns, listMethods))
D = pd.DataFrame(permutations,columns = ['Decision Pattern', 'Method'])
D['indice'] = [f"{D.iloc[i]['Decision Pattern']}_{D.iloc[i]['Method']}" for i in range(0,len(D))]




# %% define heatmap
def createHeatmap(D_scs,scs):
    D_scs = D_scs[['Decision Pattern', 'Method']]
    #D_scs = D_scs.append(D)
    D_scs = D_scs.groupby(['Decision Pattern', 'Method']).size().reset_index()
    D_scs['indice'] = [f"{D_scs.iloc[i]['Decision Pattern']}_{D_scs.iloc[i]['Method']}" for i in range(0,len(D_scs))]
    D_scs.rename(columns={0:'count'},inplace=True)
    for indice in set(D['indice']):
        if indice not in set(D_scs['indice']):
            method = D[D['indice']==indice]['Method'].iloc[0]
            pattern = D[D['indice']==indice]['Decision Pattern'].iloc[0]
            D_scs = D_scs.append(pd.DataFrame([[pattern, method,0,indice]],columns=D_scs.columns))
    #output_dataframe, output_figures = defineFromToTable(D_scs, 'Decision Pattern', 'Method', 'count')
    #define from to matrix
    D_scs_square = D_scs.pivot('Decision Pattern',columns='Method',values = 'count')
    
    #sort rows by P*
    D_scs_square['indiceSorting'] = [D_scs_square.index[i][1:] for i in range(0,len(D_scs_square))]
    D_scs_square['indiceSorting'] = D_scs_square['indiceSorting'].astype(float)
    D_scs_square.sort_values(by='indiceSorting',inplace=True)
    D_scs_square.drop(columns=['indiceSorting'],inplace=True)
    
    #plot heatmap
    plt.figure()
    sns.heatmap(D_scs_square, linewidths=.5,annot=True, fmt="d",cmap="YlOrRd")
    plt.title(scs)
    plt.savefig(f"Heatmap_{scs}.png")

#%% prepare heatmaps 

# prepare heatmaps for scss
for scs in set(D_literature['SCS']):
    D_scs = D_literature[D_literature['SCS']==scs]
    createHeatmap(D_scs,scs)


# prepare heatmaps for all
createHeatmap(D_literature,'All Supply chain Systems')

# %% create timeline

D_literature ['Decade'] = D_literature['Year']//10*10
D_years = D_literature.groupby(['Decade','Method']).size().reset_index()
D_years_square = D_years.pivot('Method',columns='Decade',values = 0)
D_years_square.drop(columns=[2020],inplace=True)

#plot heatmap
plt.figure()
sns.heatmap(D_years_square, linewidths=.5,annot=True,cmap="YlOrRd")
plt.title("Transition of the methods implementation over the time")
plt.savefig(f"timeTransition.png")
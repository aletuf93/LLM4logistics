#%% Import packages and data
import pandas as pd
import re

# %% import data
df_processed = pd.read_excel("../data/dataset_processed/export_processed_sample.xlsx")

# %% clean data
columns_to_clean_problem = ['problem_classification_phi3', 
                            'problem_classification_llama3.1',
                            'problem_classification_mistral', 
                            'problem_classification_qwen2',
                            'problem_classification_deepseek-r1:7b'
                            ]

columns_to_clean_method = ['method_classification_phi3',
                           'method_classification_llama3.1',
                           'method_classification_mistral'
                             #, 'method_classification_qwen2'
                             #,'method_classification_deepseek-r1:7b'
                             ]

def clean_think_tag(string):
    """Rimuove tutto il testo tra i tag <think> e </think>."""
    return re.sub(r'<think>.*?</think>', '', string, flags=re.DOTALL)

def clean_problem_classification(string: str):
    string = string.upper()  # everything to upper case
    if ("P1" in string[:4]):
        return "P1"
    elif ("P2" in string[:4]):
        return "P2"
    elif ("P3" in string[:4]):
        return "P3"
    elif ("P4" in string[:4]):
        return "P4"
    elif ("P5" in string[:4]):
        return "P5"
    elif ("P6" in string[:4]):
        return "P6"
    elif ("OTHER" in string[:6]):
        return "OTHER"
    
    else:  # else check the rest of the sentence
        if ("P1" in string) or ("TECHNOLOGY SELECTION PROBLEM" in string):
            return "P1"
        elif ("P2" in string) or ("LOCATION PROBLEMS" in string):
            return "P2"
        elif ("P3" in string) or ("SIZING PROBLEMS" in string):
            return "P3"
        elif ("P4" in string) or ("PLANNING PROBLEM" in string):
            return "P4"
        elif ("P5" in string) or ("QUALITY MANAGEMENT" in string):
            return "P5"
        elif ("P6" in string) or ("WORKLOAD FORECAST" in string):
            return "P6"
        else:
            return "CLASSIFICATION ERROR"
    
def clean_method_classification(string: str):
    string = string.upper()  # everything to upper case
    if ("S1" in string[:4]):
        return "S1"
    elif ("S2" in string[:4]):
        return "S2"
    elif ("S3" in string[:4]):
        return "S3"
    elif ("S4" in string[:4]):
        return "S4"
    elif ("S5" in string[:4]):
        return "S5"
    elif ("S6" in string[:4]):
        return "S6"
    elif ("S7" in string[:4]):
        return "S7"
    elif ("CT1" in string[:4]):
        return "CT1"
    elif ("OR1" in string[:4]):
        return "OR1"
    elif ("OR2" in string[:4]):
        return "OR2"
    elif ("OTHER" in string[:6]):
        return "OTHER"
    else:
        if ("S1" in string) or ("CORRELATION ANALYSIS" in string):
            return "S1"
        elif ("S2" in string) or ("STATISTICAL ANALYSIS" in string):
            return "S2"
        elif ("S3" in string) or ("BAYESIAN ANALYSIS" in string):
            return "S3"
        elif ("S4" in string) or ("SIMULATION-BASED ANALYSIS" in string):
            return "S4"
        elif ("S5" in string) or ("SUPERVISED LEARNING FOR PREDICTION" in string):
            return "S5"
        elif ("S6" in string) or ("TIME SERIES ANALYSIS" in string):
            return "S6"
        elif ("S7" in string) or ("CLUSTERING FOR CLASSIFICATION" in string):
            return "S7"
        elif ("CT1" in string) or ("ENGINEERING CONTROL FOR PREDICTION" in string):
            return "CT1"
        elif ("OR1" in string) or ("OPTIMIZATION METHODS" in string):
            return "OR1"
        elif ("OR2" in string) or ("MULTI-SCENARIO ANALYSIS" in string):
            return "OR2"
        else:
            return "CLASSIFICATION ERROR"
        
def clean_supply_chain_system(string: str):
    if (string == "production facility design") or \
        (string == "production facility control") or \
        (string == "analytics production facility"):
        return "PRODUCTION"
    elif (string == "supply chain storage inventory system design") or \
        (string == "supply chain storage inventory system control") or \
        (string == "analytics supply chain inventory storage system") or \
        (string == "storage system design")     :
        return "WAREHOUSE"
    elif (string== "supply chain distribution network design") or \
        (string == "supply chain distribution network control") or \
        (string == "analytics supply chain distribution network") or \
        (string == "distribution network control") or \
        (string == "distribution network design"):
        return "NETWORK"
    else:
        return "GENERIC"
    
def clean_analytics_family(string: str):
    '''
    try:
        cleaned_string = re.split(r'[:,]', string, 1)[1].strip()
        proc_string = cleaned_string[:min(12, len(cleaned_string) )]
    except:
        proc_string = string
    '''
    if "DESCRIPTIVE" in string:
        return "DESCRIPTIVE"
    elif "PREDICTIVE" in string:
        return "PREDICTIVE"
    elif "PRESCRIPTIVE" in string:
        return "PRESCRIPTIVE"
    else:
        return "CLASSIFICATION ERROR"

df_cleaned = df_processed

#Clean think tag for deepseek
for column in ['problem_classification_deepseek-r1:7b']:
   df_processed[column] = [clean_think_tag(str(i)) for i in df_processed[column]]
 

#Clean Problem
for column in columns_to_clean_problem:
    df_cleaned[f"{column}_cleaned"] = [clean_problem_classification(str(i)) for i in df_processed[column]]

#Clean Method
for column in columns_to_clean_method:
    df_cleaned[f"{column}_cleaned"] = [clean_method_classification(str(i)) for i in df_processed[column]]

#Clean Analytics Family
for column in columns_to_clean_method:
    df_cleaned[f"{column}_AnalyticsFamily"] = [clean_analytics_family(str(i)) for i in df_processed[column]]

#Clean Supply Chain System
df_cleaned["Supply chain System"] = [clean_supply_chain_system(str(i)) for i in df_processed["Query"]]


df_cleaned.to_excel("../data/dataset_processed/export_cleaned.xlsx")



# %%

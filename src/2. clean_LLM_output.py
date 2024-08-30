#%% Import packages and data
import pandas as pd

# %% import data
df_processed = pd.read_excel("../data/dataset_processed/export_processed.xlsx")

# %% clean data
columns_to_clean_problem = ['problem_classification_phi3', 'problem_classification_llama3.1',
                    'problem_classification_mistral', 'problem_classification_qwen2']

columns_to_clean_method = ['method_classification_phi3', 'method_classification_llama3.1',
                            'method_classification_mistral', 'method_classification_qwen2']


def clean_problem_classification(string: str):
    string = string.upper()  # everything to upper case
    if ("P10" in string[:4]):  # first check at the beginning of the sentence
        return "P10"
    elif ("P1" in string[:4]):
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
    elif ("P7" in string[:4]):
        return "P7"
    elif ("P8" in string[:4]):
        return "P8"
    elif ("P9" in string[:4]):
        return "P9"
    elif ("OTHER" in string[:6]):
        return "OTHER"
    
    else:  # else check the rest of the sentence
        if ("P10" in string) or ("OPERATIONS MANAGEMENT" in string):
            return "P10"
        elif ("P1" in string) or ("FAMILY PROBLEMS" in string):
            return "P1"
        elif ("P2" in string) or ("ASSIGNMENT PROBLEMS" in string):
            return "P2"
        elif ("P3" in string) or ("FLOW PROBLEMS" in string):
            return "P3"
        elif ("P4" in string) or ("MECHANICAL PLANT AND EQUIPMENT DESIGN" in string):
            return "P4"
        elif ("P5" in string) or ("POWER PROBLEMS" in string):
            return "P5"
        elif ("P6" in string) or ("PLACEMENT PROBLEMS" in string):
            return "P6"
        elif ("P7" in string) or ("DISPATCHING RULES" in string):
            return "P7"
        elif ("P8" in string) or ("PERFORMANCE ASSESSMENT" in string):
            return "P8"
        elif ("P9" in string) or ("WORKLOAD PREDICTION" in string):
            return "P9"
        else:
            return "CLASSIFICATION ERROR"
    
def clean_method_classification(string: str):
    string = string.upper()  # everything to upper case
    if ("PD1" in string[:4]):
        return "PD1"
    elif ("PD2" in string[:4]):
        return "PD2"
    elif ("PD3" in string[:4]):
        return "PD3"
    elif ("D1" in string[:4]):
        return "D1"
    elif ("D2" in string[:4]):
        return "D2"
    elif ("D3" in string[:4]):
        return "D3"
    elif ("D4" in string[:4]):
        return "D4"
    elif ("PS1" in string[:4]):
        return "PS1"
    elif ("PS2" in string[:4]):
        return "PS2"
    elif ("PS3" in string[:4]):
        return "PS3"
    elif ("PS4" in string[:4]):
        return "PS4"
    elif ("OTHER" in string[:6]):
        return "OTHER"
    else:
        if ("PD1" in string) or ("SUPERVISED LEARNING FOR PREDICTION" in string):
            return "PD1"
        elif ("PD2" in string) or ("TIME SERIES ANALYSIS" in string):
            return "PD2"
        elif ("PD3" in string) or ("ENGINEERING CONTROL FOR PREDICTION" in string):
            return "PD3"
        elif ("D1" in string) or ("CORRELATION ANALYSIS" in string):
            return "D1"
        elif ("D2" in string) or ("STATISTICAL ANALYSIS" in string):
            return "D2"
        elif ("D3" in string) or ("BAYESIAN ANALYSIS" in string):
            return "D3"
        elif ("D4" in string) or ("SIMULATION-BASED ANALYSIS" in string):
            return "D4"
        elif ("PS1" in string) or ("OPTIMIZATION METHODS" in string):
            return "PS1"
        elif ("PS2" in string) or ("CLUSTERING FOR CLASSIFICATION" in string):
            return "PS2"
        elif ("PS3" in string) or ("MULTI-SCENARIO ANALYSIS" in string):
            return "PS3"
        elif ("PS4" in string) or ("ENGINEERING CONTROL FOR PRESCRIPTION" in string):
            return "PS4"
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

df_cleaned = df_processed

for column in columns_to_clean_problem:
    df_cleaned[f"{column}_cleaned"] = [clean_problem_classification(i) for i in df_processed[column]]

for column in columns_to_clean_method:
    df_cleaned[f"{column}_cleaned"] = [clean_method_classification(i) for i in df_processed[column]]

df_cleaned["Supply chain System"] = [clean_supply_chain_system(i) for i in df_processed["Query"]]

df_cleaned.to_excel("../data/dataset_processed/export_cleaned.xlsx")



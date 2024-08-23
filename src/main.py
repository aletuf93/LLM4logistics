#%% Import packages
import ollama
import pandas as pd
import re
import datetime
from yaml_replace import YAMLTemplate

# Define the models to test
model_list = ["phi3",       #3B parameters
              "llama3.1",   #8B parameters
              "mistral",    #7B parameters
              "qwen2"]      #7B parameters

problem_list = ["problem_classification", "method_classification"]

def test_query_LLM(model, prompt):
    return f"{model}__{prompt}"


# Define functions
def query_LLM(model, prompt):
    client = ollama.Client(host='http://localhost:11434')
    try:
        answer = client.generate(model=model,
                                   prompt=prompt)
        response = answer["response"]
    except Exception as error:
        response = str(error)
    return response



def clean_string(string):
    string = re.sub(r'[^A-Za-z0-9 ]', ' ', string)
    return string


def generate_prompt(title, abstract, problem):

    # clean strings
    title = clean_string(title)
    abstract = clean_string(abstract)



    dict_prompt = YAMLTemplate.from_file('../config/prompt.yaml').render(
        {
            'title': f"""{title}""",
            'abstract': f"""{abstract}""",
        }
    )

    prompt = dict_prompt[problem]
    return prompt


#%% Import data
df = pd.read_excel("../data/_master_analysis.xlsx")

# %% classify problems
for problem in problem_list:
    for model in model_list:
        output = []
        for i in range(0, len(df)):
            print(f"{datetime.datetime.now()} | problem: {problem}; model: {model}; index:{i}/{len(df)-1}")
            title = df["Title"].iloc[i]
            abstract = df["Abstract"].iloc[i]

            # Generate prompt
            prompt = generate_prompt(title, abstract, problem)

            # Query LLM
            result = test_query_LLM(model, prompt)

            # append result
            output.append(result)
        df[f"{problem}_{model}"] = output
        df.to_excel("export.xlsx") # save intermediate results


# %%

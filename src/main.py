#%% Import packages
import ollama
from yaml_replace import YAMLTemplate

# %% Set the model list
model_list = ["phi3",       #3B parameters
              "llama3.1",   #8B parameters
              "mistral",    #7B parameters
              "qwen2"]      #7B parameters
# %% Import prompt
title = "Big data analytics in logistics and supply chain management: Certain investigations for research and applications"
abstract = "The amount of data produced and communicated over the Internet is significantly increasing, thereby creating challenges for the organizations that would like to reap the benefits from analyzing this massive influx of big data. This is because big data can provide unique insights into, inter alia, market trends, customer buying patterns, and maintenance cycles, as well as into ways of lowering costs and enabling more targeted business decisions. Realizing the importance of big data business analytics (BDBA), we review and classify the literature on the application of BDBA on logistics and supply chain management (LSCM) – that we define as supply chain analytics (SCA), based on the nature of analytics (descriptive, predictive, prescriptive) and the focus of the LSCM (strategy and operations). To assess the extent to which SCA is applied within LSCM, we propose a maturity framework of SCA, based on four capability levels, that is, functional, process-based, collaborative, agile SCA, and sustainable SCA. We highlight the role of SCA in LSCM and denote the use of methodologies and techniques to collect, disseminate, analyze, and use big data driven information. Furthermore, we stress the need for managers to understand BDBA and SCA as strategic assets that should be integrated across business activities to enable integrated enterprise business analytics. Finally, we outline the limitations of our study and future research directions."

dict_prompt = YAMLTemplate.from_file('../config/prompt.yaml').render(
    {
        'title': title,
        'abstract': abstract,
    }
)

prompt_problem_classification = dict_prompt["problem_classification"]
# %% Connect to the local client
client = ollama.Client(host='http://localhost:11434')

# %% generate a response
response = client.generate(model=model_list[0],
                           prompt=prompt_problem_classification)

print(response["response"])
# %%

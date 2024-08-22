#%% Import packages
import ollama

# %% Set the model list
model_list = ["phi3",       #3B parameters
              "llama3.1",   #8B parameters
              "mistral",    #7B parameters
              "qwen2"]      #7B parameters

# %% Connecti to the local client
client = ollama.Client(host='http://localhost:11434')

# %% generate a response
response = client.generate(model=model_list[0],
                           prompt="Answer with a single word. can you speak Italian?")

print(response["response"])
# %%

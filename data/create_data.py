import requests
import json

response = requests.get("https://west.ucsd.edu/llm_project/?endpoint=get_targets")
data = response.json()

list = []
for protein in data:
    list.append({"instruction": protein, "input": "", "output": ""})
print(len(data))

# with open("./data/proteins.json", "w") as file:
#     json.dump(list, file, indent=4)
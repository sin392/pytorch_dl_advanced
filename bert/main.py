# %%
from addict import Dict
import json

config_file = "./weights/bert_config.json"
json_file = open(config_file, "r")
config = json.load(json_file)
# %%
print(config)

# %%

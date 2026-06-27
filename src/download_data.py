import os

import yaml
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

api_key = os.environ["ROBOFLOW_API_KEY"]
rf = Roboflow(api_key=api_key)

project = rf.workspace("research-new-things-m0fiq").project("microplastic-v2-wowak")
dataset = project.version(1).download("yolov11", location="data", overwrite=True)

data_yaml_path = os.path.join(dataset.location, "data.yaml")
with open(data_yaml_path) as f:
    data_yaml = yaml.safe_load(f)

print(f"data.yaml path: {data_yaml_path}")
print(f"Class names: {data_yaml['names']}")

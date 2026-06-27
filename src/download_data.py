import os

from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

api_key = os.environ["ROBOFLOW_API_KEY"]
rf = Roboflow(api_key=api_key)

# Paste your Roboflow project details here, e.g.:
# project = rf.workspace("your-workspace").project("your-project")
# dataset = project.version(1).download("yolov11", location="data")

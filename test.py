import torch
import os
from roboflow import Roboflow

print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

rf = Roboflow(api_key="Iw97CqX7K9hULjLXEF5l")
project = rf.workspace("carth").project("car_detection-5uyan")
version = project.version(7)
dataset = version.download("yolov5")

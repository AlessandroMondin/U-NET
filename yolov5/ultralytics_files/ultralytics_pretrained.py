import matplotlib.pyplot as plt
import torch
#from model import check_size

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')  # or yolov5n - yolov5x6, custom

#check_size(model)

# Images
img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
#results = model(img)

# Results
#results.show()
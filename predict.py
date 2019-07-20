import helpers.NeuralNetUtils
import torchvision
from torchvision import datasets, transforms, models
import helpers.LoadJson
import argparse


args_parser = argparse.ArgumentParser(description='predict-file')
args_parser.add_argument('input_img', default='./flowers/test/1/image_06743.jpg', nargs='*', action="store", type = str)
args_parser.add_argument('checkpoint', default='checkpoint.pth', nargs='*', action="store",type = str)
args_parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
# args_parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
args_parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

args = args_parser.parse_args()
path_image = args.input_img
top_k = args.top_k
gpu_or_cpu = args.gpu
input_img = args.input_img
checkpoint_path = args.checkpoint


# Load the model
model = helpers.NeuralNetUtils.load_model(checkpoint_path)

# Load the content of the json file
categories = helpers.JsonLoader.load_json()

# Predict
helpers.NeuralNetUtils.predict(input_img, model, top_k)
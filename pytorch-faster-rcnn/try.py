import torchvision
import torch
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
training_video_nums = 447
video_frames = 30
feature_num = 100
# Training data would be something with size [video_frames, training_video_nums, features_num]
# For processing the images, we are going to use pretrained model. Thus we can view the input of the network as an array of
# training_video_nums * video_frames * features_num
def getVideoFeatures(video_dir):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, num_classes=91)
    model.cuda()
    model.eval()
    video_features = torch.rand(training_video_nums, video_frames, feature_num)    
    videos =  next(os.walk(video_dir))[1]
    for i in range(len(videos)):
        print(videos[i])
        frames = next(os.walk(video_dir + "/" + videos[i]))[2]
        for j in range(len(frames)):
            image_path = video_dir + "/" + videos[i] + "/" + frames[j]
            _,ext = os.path.splitext(image_path)
            if ext != '.jpg':
                continue
            tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
            img = tfms(Image.open(image_path)).unsqueeze(0)
            #print(img.shape)
            predictions = model(img.to(device))
            image_feature = predictions[0]["boxes"].detach().reshape(-1)
            video_features[i][j][:len(image_feature)] = image_feature.to("cpu")
        np.savetxt(video_dir + "/" + videos[i] + "/" + "output.csv", video_features[i], delimiter=',')

def getVideoFeaturesFromCsv(video_dir):
    all_video_features = torch.rand(training_video_nums, video_frames, feature_num, device=device)
    video_features = torch.rand(training_video_nums, video_frames, feature_num)
    videos =  next(os.walk(video_dir))[1]
    for i in range(1):
        print(videos[i])
        video_features_file = video_dir + "/" + videos[i] + "/" "output.csv"
        print(video_features_file)
        video_features = torch.tensor(pd.read_csv(video_features_file, header=None).values)
        all_video_features[i] = video_features
    print(all_video_features)
#getVideoFeatures("../train/train")
getVideoFeaturesFromCsv("../train/train")
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, num_classes=91)
# # # For training
# # images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
# # labels = torch.randint(1, 91, (4, 11))
# # images = list(image for image in images)
# # targets = []
# # for i in range(len(images)):
# #     d = {}
# #     d['boxes'] = boxes[i]
# #     d['labels'] = labels[i]
# #     targets.append(d)
# # output = model(images, targets)
# # For inference
# # Preprocess image
# tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
# img = tfms(Image.open('img.jpg')).unsqueeze(0)
# print(img.shape) # torch.Size([1, 3, 224, 224])

# model.eval()
# x = img
# predictions = model(x)
# print(predictions)
# optionally, if you want to export the model to ONNX:
#torch.onnx.export(model, x, "faster_rcnn.onnx", opset_version = 11)


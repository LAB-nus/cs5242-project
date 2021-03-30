from parseJson import parseProjectJson
import json
from PIL import Image
import torch
import os
from torchvision import transforms
import numpy as np
from efficientnet_pytorch import EfficientNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def main():
    getVideoFeatures("train/train")

def getVideoFeatures(video_dir):
    training_video_nums = 447
    video_frames = 30
    feature_num = 143360

    objData, relData = parseProjectJson()
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model.cuda()
    model.eval()
    # Preprocess image
    tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

    video_features = torch.rand(training_video_nums, video_frames, feature_num)
    # Load class names
    swappedObjData = dict([(value, key) for key, value in objData.items()])
    swappedRelData = dict([(value, key) for key, value in relData.items()])
    labels_map = np.array([swappedObjData[i] for i in range(34)])
    labels_map = torch.tensor(np.arange(34))
    print(labels_map.shape)

    videos =  next(os.walk(video_dir))[1]
    for i in range(len(videos)):
        frames = next(os.walk(video_dir + "/" + videos[i]))[2]
        for j in range(len(frames)):
            image_path = video_dir + "/" + videos[i] + "/" + frames[j]
            _,ext = os.path.splitext(image_path)
            if ext != '.jpg':
                continue
            tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
            img = tfms(Image.open(image_path)).unsqueeze(0)
            
            features = model.extract_features(img.to(device))
            video_features[i][j][:] = torch.flatten(features).to("cpu")

        torch.detach().numpy().savetxt(video_dir + "/" + videos[i] + "/" + "output.csv", video_features[i])

            #print(image_feature.shape)
        # for j in range(len(frames)):
        #     image_path = video_dir + "/" + videos[i] + "/" + frames[j]
        #     tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        #     img = tfms(Image.open(image_path))
        #     imgs[j] = img
        # for j in range(len(frames)):
        #     print(imgs.shape)
        #     with torch.no_grad():
        #         predictions = model(imgs)
        #         image_feature = predictions[0]["boxes"].reshape(-1)
        #         print(image_feature.shape)

if __name__=="__main__":
    main()
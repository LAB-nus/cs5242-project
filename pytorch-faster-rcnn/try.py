import torchvision
import torch
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
training_video_nums = 447
video_frames = 30
feature_num = 100
word_vec_len = 10
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

word_to_idx_1 = {"turtle": 0, "antelope": 1, "bicycle": 2, "lion": 3, "ball": 17, "motorcycle": 5, "cattle": 6, "airplane": 7, "car": 33, "sheep": 8, "horse": 9, "watercraft": 10, "monkey": 11, "fox": 12, "giant_panda": 13, "elephant": 14, "bird": 15, "domestic_cat": 32, "frisbee": 4, "squirrel": 18, "bus": 19, "bear": 21, "tiger": 22, "train": 23, "snake": 24, "rabbit": 25, "whale": 26, "red_panda": 20, "skateboard": 28, "dog": 29, "person": 30, "lizard": 31, "hamster": 16, "sofa": 27, "zebra": 34}
word_to_idx_2 = {"fly_next_to": 0, "run_with": 1, "left": 2, "lie_behind": 3, "run_above": 4, "jump_toward": 5, "jump_front": 6, "run_behind": 7, "run_front": 8, "walk_past": 9, "sit_left": 10, "stop_left": 11, "right": 12, "move_front": 13, "swim_right": 14, "lie_left": 15, "walk_beneath": 16, "walk_behind": 17, "stop_right": 18, "sit_next_to": 19, "run_next_to": 20, "creep_right": 21, "move_past": 22, "swim_left": 23, "move_beneath": 24, "fly_above": 25, "move_left": 26, "above": 27, "bite": 28, "beneath": 29, "move_behind": 30, "lie_above": 31, "run_left": 32, "move_toward": 33, "next_to": 34, "stand_next_to": 35, "creep_behind": 36, "jump_right": 37, "walk_right": 38, "walk_above": 39, "stand_above": 40, "creep_front": 41, "stand_front": 42, "taller": 43, "stop_beneath": 44, "watch": 45, "jump_beneath": 46, "stand_behind": 47, "lie_next_to": 48, "sit_above": 49, "lie_right": 50, "play": 51, "larger": 52, "sit_inside": 53, "swim_next_to": 54, "sit_behind": 55, "jump_left": 56, "walk_left": 57, "fly_away": 58, "stop_front": 59, "sit_beneath": 60, "creep_left": 61, "move_with": 62, "stand_right": 63, "lie_front": 64, "walk_front": 65, "run_beneath": 66, "behind": 67, "sit_front": 68, "jump_past": 69, "run_right": 70, "walk_toward": 71, "run_past": 72, "front": 73, "sit_right": 74, "stand_left": 75, "jump_behind": 76, "swim_front": 77, "move_right": 78, "walk_next_to": 79, "swim_behind": 80, "stop_behind": 81}
word_to_idx = {}
for word in word_to_idx_1:
    word_to_idx[word] = word_to_idx_1[word]
for word in word_to_idx_2:
    word_to_idx[word] = word_to_idx_2[word] + len(word_to_idx_1)
embeds = nn.Embedding(len(word_to_idx), word_vec_len)  

def getVideoFeaturesFromCsv(video_dir):
    all_video_features = torch.rand(training_video_nums, video_frames, feature_num+3, device=device) # Plus 3 as we have three extra sequence for output
    videos =  next(os.walk(video_dir))[1]
    for i in range(1):
        video_features_file = video_dir + "/" + videos[i] + "/" "output.csv"
        video_features = torch.tensor(pd.read_csv(video_features_file, header=None).values)
        all_video_features[i][:len(video_features)] = video_features
    return all_video_features

class LSTM(nn.Module):
    def __init__(self, seq_num=33, feature_num=100, obj_label_num=35, relation_label_num=82, hidden_layer_size_1=30, hidden_layer_size_2=30, batch_size=20):
        super().__init__()
        self.batch_size = batch_size
        self.seq_num = seq_num
        self.lstm_1 = nn.LSTM(seq_num, feature_num)
        self.lstm_2 = []
        self.hidden_layer_size_2 = hidden_layer_size_2
        for i in range(seq_num):
            self.lstm_2.append(nn.LSTMCell(hidden_layer_size_1+word_vec_len, hidden_layer_size_2))
        print(self.lstm_2)
        self.linear_obj_1 = nn.Linear(hidden_layer_size_2, obj_label_num)
        self.linear_obj_2 = nn.Linear(hidden_layer_size_2word_vec_len, obj_label_num)
        self.linear_relation = nn.Linear(hidden_layer_size_2, relation_label_num)
        self.hidden_cell_1 = (torch.zeros(1, batch_size, hidden_layer_size_1),
                            torch.zeros(1, batch_size, hidden_layer_size_1))
        self.hidden_cell_2 = (torch.zeros(1, batch_size, hidden_layer_size_2),
                            torch.zeros(1, batch_size, hidden_layer_size_2))

    def forward(self, input):
        _, batch, _ = input.shape
        lstm_out_1, _ = self.lstm(input, self.hidden_cell_1) #  (seq_len, batch, num_directions * hidden_size)
        hidden_cell_input_2 = self.hidden_cell_2
        lstm_out_2 = torch.zeros()
        word_vec = tensor.zeros(batch, word_vec_len)
        for i in range(seq_num-2):
            input_for_cell = torch.cat((lstm_out_1[i], word_vec), 1)            
            lstm_out_2 = self.lstm_2[i](input_for_cell, lstm_out_2)
        
        out = []

        # For the last two words, we would like to use the word predicted previously.
        hx, _ = lstm_out_2
        obj_1_prediction = self.linear_obj_1(hx) # batch * obj_label_num
        out.append(obj_1_prediction)
        chosen = torch.argmax(obj_1_prediction, dim=1)
        word_vec = embeds(chosen)

        input_for_cell = torch.cat((lstm_out_1[seq_num-2], word_vec), 1)            
        lstm_out_2 = self.lstm_2[i](input_for_cell, lstm_out_2)
        hx, _ = lstm_out_2
        obj_2_prediction = self.linear_obj_2(hx) # batch * obj_label_num
        out.append(obj_1_prediction)
        chosen = torch.argmax(obj_1_prediction, dim=1)
        word_vec = embeds(chosen)

        input_for_cell = torch.cat((lstm_out_1[seq_num-1], word_vec), 1)            
        lstm_out_2 = self.lstm_2[i](input_for_cell, lstm_out_2)
        hx, _ = lstm_out_2
        relation_prediction = self.linear_obj_2(hx) # batch * obj_label_num
        out.append(relation_prediction)

        return out

#getVideoFeatures("../train/train")
video_input = getVideoFeaturesFromCsv("../train/train")
print(video_input.shape)
LSTM()
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


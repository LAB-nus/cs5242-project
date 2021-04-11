import torchvision
import torch
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd
import torch.nn as nn
import json
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
feature_file_name = "efficient_net_feature_large_2.csv"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
training_video_nums = 448
test_video_nums = 128
video_frames = 30
feature_num = 20480
word_vec_len = 100
# Training data would be something with size [video_frames, training_video_nums, features_num]
# For processing the images, we are going to use pretrained model. Thus we can view the input of the network as an array of
# training_video_nums * video_frames * features_num
def getVideoFeatures(video_dir, mode):
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model.cuda()
    model.eval()
    if mode == 'training':
        video_nums = training_video_nums
    elif mode == 'testing':
        video_nums = test_video_nums

    video_features = torch.zeros(video_nums, video_frames, feature_num)    
    videos =  next(os.walk(video_dir))[1]
    for i in range(len(videos)):
        print(videos[i])
        frames = next(os.walk(video_dir + "/" + videos[i]))[2]
        for j in range(len(frames)):
            image_path = video_dir + "/" + videos[i] + "/" + frames[j]
            _,ext = os.path.splitext(image_path)
            if ext != '.jpg':
                continue
            tfms = transforms.Compose([transforms.Resize(1280), transforms.CenterCrop(1280), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
            img = tfms(Image.open(image_path)).unsqueeze(0)
            features = model.extract_features(img.to(device)).detach()
            compressed_features = compress(features)
            print(compressed_features.shape)
            video_features[i][j][:len(compressed_features)] = compressed_features.to("cpu")
        np.savetxt(video_dir + "/" + videos[i] + "/" + feature_file_name, video_features[i], delimiter=',')

def compress(img_feature):
    m = nn.AvgPool2d((7, 7), stride=(7, 7), padding=1)
    pooled_feature = m(img_feature)
    return pooled_feature.reshape(-1)

word_to_idx_1 = {"turtle": 0, "antelope": 1, "bicycle": 2, "lion": 3, "ball": 17, "motorcycle": 5, "cattle": 6, "airplane": 7, "car": 33, "sheep": 8, "horse": 9, "watercraft": 10, "monkey": 11, "fox": 12, "giant_panda": 13, "elephant": 14, "bird": 15, "domestic_cat": 32, "frisbee": 4, "squirrel": 18, "bus": 19, "bear": 21, "tiger": 22, "train": 23, "snake": 24, "rabbit": 25, "whale": 26, "red_panda": 20, "skateboard": 28, "dog": 29, "person": 30, "lizard": 31, "hamster": 16, "sofa": 27, "zebra": 34}
word_to_idx_2 = {"fly_next_to": 0, "run_with": 1, "left": 2, "lie_behind": 3, "run_above": 4, "jump_toward": 5, "jump_front": 6, "run_behind": 7, "run_front": 8, "walk_past": 9, "sit_left": 10, "stop_left": 11, "right": 12, "move_front": 13, "swim_right": 14, "lie_left": 15, "walk_beneath": 16, "walk_behind": 17, "stop_right": 18, "sit_next_to": 19, "run_next_to": 20, "creep_right": 21, "move_past": 22, "swim_left": 23, "move_beneath": 24, "fly_above": 25, "move_left": 26, "above": 27, "bite": 28, "beneath": 29, "move_behind": 30, "lie_above": 31, "run_left": 32, "move_toward": 33, "next_to": 34, "stand_next_to": 35, "creep_behind": 36, "jump_right": 37, "walk_right": 38, "walk_above": 39, "stand_above": 40, "creep_front": 41, "stand_front": 42, "taller": 43, "stop_beneath": 44, "watch": 45, "jump_beneath": 46, "stand_behind": 47, "lie_next_to": 48, "sit_above": 49, "lie_right": 50, "play": 51, "larger": 52, "sit_inside": 53, "swim_next_to": 54, "sit_behind": 55, "jump_left": 56, "walk_left": 57, "fly_away": 58, "stop_front": 59, "sit_beneath": 60, "creep_left": 61, "move_with": 62, "stand_right": 63, "lie_front": 64, "walk_front": 65, "run_beneath": 66, "behind": 67, "sit_front": 68, "jump_past": 69, "run_right": 70, "walk_toward": 71, "run_past": 72, "front": 73, "sit_right": 74, "stand_left": 75, "jump_behind": 76, "swim_front": 77, "move_right": 78, "walk_next_to": 79, "swim_behind": 80, "stop_behind": 81}
word_to_idx = {}
for word in word_to_idx_1:
    word_to_idx[word] = word_to_idx_1[word]
for word in word_to_idx_2:
    word_to_idx[word] = word_to_idx_2[word] + len(word_to_idx_1)
embeds = nn.Embedding(len(word_to_idx), word_vec_len).to(device)  

def getVideoFeaturesFromCsv(video_dir, mode):
    video_nums = 0
    if mode == 'training':
        video_nums = training_video_nums
    elif mode == 'testing':
        video_nums = test_video_nums

    all_video_features = torch.zeros(video_nums, video_frames+3, feature_num, device=device) # Plus 3 as we have three extra sequence for output
    videos =  next(os.walk(video_dir))[1]
    for i in range(video_nums):
        print(i)
        video_features_file = video_dir + "/" + videos[i] + "/" + feature_file_name
        print(video_features_file)
        video_features = torch.tensor(pd.read_csv(video_features_file, header=None).values)
        all_video_features[i][:-3] = video_features
    return all_video_features

class LSTM(nn.Module):
    def __init__(self, seq_num=33, feature_num=20480, obj_label_num=35, relation_label_num=82, hidden_layer_size_1=900, hidden_layer_size_2=900, batch_size=2):
        super().__init__()
        self.batch_size = batch_size
        self.seq_num = seq_num
        self.lstm_1 = nn.LSTM(feature_num, hidden_layer_size_1).to(device)
        self.lstm_2 = []
        self.hidden_layer_size_2 = hidden_layer_size_2
        self.hidden_layer_size_2 = hidden_layer_size_2
        for i in range(seq_num):
            self.lstm_2.append(nn.LSTMCell(hidden_layer_size_1+word_vec_len, hidden_layer_size_2).to(device))
        #print(self.lstm_2)
        self.linear_obj_1 = nn.Linear(hidden_layer_size_2, obj_label_num)
        self.linear_obj_2 = nn.Linear(hidden_layer_size_2, obj_label_num)
        self.linear_relation = nn.Linear(hidden_layer_size_2, relation_label_num)
        self.hidden_cell_1 = (torch.zeros(1, batch_size, hidden_layer_size_1, device=device),
                            torch.zeros(1, batch_size, hidden_layer_size_1, device=device))
        self.hidden_cell_2 = (torch.zeros(batch_size, hidden_layer_size_2, device=device),
                            torch.zeros(batch_size, hidden_layer_size_2, device=device))

    def forward(self, input):
        _, batch, _ = input.shape
        lstm_out_1, _ = self.lstm_1(input, self.hidden_cell_1) #  (seq_len, batch, num_directions * hidden_size)
        lstm_out_2 = self.hidden_cell_2
        word_vec = torch.zeros(batch, word_vec_len, device=device)
        for i in range(self.seq_num-2):
            input_for_cell = torch.cat((lstm_out_1[i], word_vec), 1)
            lstm_out_2 = self.lstm_2[i](input_for_cell, lstm_out_2)
        
        out = []

        # For the last two words, we would like to use the word predicted previously.
        hx, _ = lstm_out_2
        obj_1_prediction = self.linear_obj_1(hx) # batch * obj_label_num
        out.append(obj_1_prediction)
        chosen = torch.argmax(obj_1_prediction, dim=1)
        word_vec = embeds(chosen)

        input_for_cell = torch.cat((lstm_out_1[self.seq_num-2], word_vec), 1)            
        lstm_out_2 = self.lstm_2[self.seq_num-2](input_for_cell, lstm_out_2)
        hx, _ = lstm_out_2
        relation_prediction = self.linear_relation(hx) # batch * obj_label_num
        out.append(relation_prediction)
        chosen = torch.argmax(relation_prediction, dim=1)
        word_vec = embeds(chosen+35)

        input_for_cell = torch.cat((lstm_out_1[self.seq_num-1], word_vec), 1)            
        lstm_out_2 = self.lstm_2[self.seq_num-1](input_for_cell, lstm_out_2)
        hx, _ = lstm_out_2
        obj_2_prediction = self.linear_obj_2(hx) # batch * obj_label_num
        out.append(obj_2_prediction)

        return out

def getTargets(batch_size, file_name='../training_annotation.json'):
    json_target = json.load(open(file_name))
    obj_1_target = torch.zeros(batch_size, device=device, dtype=int)
    obj_2_target = torch.zeros(batch_size, device=device, dtype=int)
    relation_target = torch.zeros(batch_size, device=device, dtype=int)
    idx = 0
    for _, target in json_target.items():
        obj_1_target[idx] = target[0]
        obj_2_target[idx] = target[2]
        relation_target[idx] = target[1]
        idx += 1
    return obj_1_target, relation_target, obj_2_target

def getScore(prob, target):
    score = .0
    batch, _ = prob.shape
    for i in range(batch):
        if target[i] == torch.argmax(prob[i]):
            score+=1
    return score / batch

def train(video_input_training):
    model = LSTM()
    model.to(device)
    obj_1_target, relation_target, obj_2_target = getTargets(448, file_name='../training_annotation.json')

    loss_function = nn.NLLLoss()
    m = nn.LogSoftmax(dim=1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    dataset = TensorDataset(video_input_training, obj_1_target, relation_target, obj_2_target)
    train_loader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)

    for epoch in range(30000):
        total_loss = 0
        for i, data  in enumerate(train_loader):
            input, target_obj_1, target_relation, target_obj_2 = data
            input = torch.transpose(input, 1, 0)
            model.zero_grad()
            out = model(input)
            loss_obj_1 = loss_function(m(out[0]), target_obj_1)
            loss_relation = loss_function(m(out[1]), target_relation)
            loss_obj_2 = loss_function(m(out[2]), target_obj_2)
            loss = loss_obj_1 +loss_relation + loss_obj_2
            total_loss += loss
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            print("epoch:", epoch, "loss", total_loss, "score_obj1", getScore(out[0], target_obj_1), "score_relation", getScore(out[1], target_relation), "score_obj2", getScore(out[2], target_obj_2))
        if epoch > 0 and epoch % 300 == 0:
            model_name = "48_model_after_"+str(epoch)
            torch.save(model, model_name)
        # if epoch > 0 and epoch % 300 == 0:
        #     test(str(epoch), model_name)

def get_top_5(prob):
    chosen = torch.topk(prob, 5).indices
    print("debugchosen")
    print(prob)
    print(chosen)
    res = ""
    for i in range(5):
        res = res + str(chosen[i].item())
        if i != 4:
            res = res + " "
    print(res)
    return res
        
def test(epoch, model_name):
    model = torch.load(model_name)
    model.to(device)
    model.eval()
    prediction = []
    prediction.append(["Id", "label"])
    probs = []
    model.zero_grad()
    id = 1
    outputFileName = epoch + '.csv'
    for i in range(64):
        batch_input = video_input_testing[i*2:i*2+2]
        batch_input = torch.transpose(batch_input, 1, 0)
        obj_1, relation, obj_2 = model(batch_input)
        for j in range(2):
            if id > 119 * 3:
                break
            prediction.append([str(id-1), get_top_5(obj_1[j])])
            id += 1            
            prediction.append([str(id-1), get_top_5(relation[j])])
            id += 1            
            prediction.append([str(id-1), get_top_5(obj_2[j])])
            id += 1            
    print(prediction)
    np.savetxt(outputFileName, prediction, delimiter=',', fmt="%s")
#getVideoFeatures("../train/train", 'training')
#getVideoFeatures("../test/test", 'testing')
#video_input_training = getVideoFeaturesFromCsv("../train/train", 'training')
video_input_testing = getVideoFeaturesFromCsv("../test/test", 'testing')
#train(video_input_training)
test("48_300", "48_model_after_300")

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
    if mode == 'trainig':
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
            tfms = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
            img = tfms(Image.open(image_path)).unsqueeze(0)
            # predictions = model(img.to(device))
            features = model.extract_features(img.to(device)).detach()
            compressed_features = compress(features)
            print(compressed_features.shape)
        #     image_feature = predictions[0]["boxes"].detach().reshape(-1)
            video_features[i][j][:len(compressed_features)] = compressed_features.to("cpu")
        np.savetxt(video_dir + "/" + videos[i] + "/" + feature_file_name, video_features[i], delimiter=',')

def compress(img_feature):
    m = nn.AvgPool2d((2, 2), stride=(2, 2), padding=1)
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
        video_features = torch.tensor(pd.read_csv(video_features_file, header=None).values)
        all_video_features[i][:-3] = video_features
    return all_video_features

class LSTM(nn.Module):
    def __init__(self, seq_num=33, feature_num=20480, obj_label_num=35, relation_label_num=82, hidden_layer_size_1=200, hidden_layer_size_2=200, batch_size=32):
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
        #print(input.shape)
        _, batch, _ = input.shape
        lstm_out_1, _ = self.lstm_1(input, self.hidden_cell_1) #  (seq_len, batch, num_directions * hidden_size)
        lstm_out_2 = self.hidden_cell_2
        word_vec = torch.zeros(batch, word_vec_len, device=device)
        for i in range(self.seq_num-2):
            input_for_cell = torch.cat((lstm_out_1[i], word_vec), 1)
            #print("debughere")
            #print(input_for_cell.shape)
            lstm_out_2 = self.lstm_2[i](input_for_cell, lstm_out_2)
        
        out = []

        # For the last two words, we would like to use the word predicted previously.
        hx, _ = lstm_out_2
        #print(hx.shape)
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

def getTargets(batch_size):
    json_target = json.load(open('../training_annotation.json'))
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
    score_map = [1, 0.5, 0.33, 0.25, 0.2]
    score = .0
    batch, _ = prob.shape
    for i in range(batch):
        rank = 0
        for j in range(len(prob[i])):
            if prob[i][j] > prob[i][target[i]]:
                rank+=1
        if rank < 5:
            score += score_map[rank]
    return score / batch

def train():
    video_input = getVideoFeaturesFromCsv("../train/train", 'training')
    #video_input = torch.transpose(video_input, 1,0)
    #print(video_input.shape)
    model = LSTM()
    model.to(device)
    obj_1_target, relation_target, obj_2_target = getTargets(448)
    loss_function = nn.NLLLoss()
    m = nn.LogSoftmax(dim=1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    dataset = TensorDataset(video_input, obj_1_target, relation_target, obj_2_target)
    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    for epoch in range(30000):  # again, normally you would NOT do 300 epochs, it is toy data
        total_loss = 0
        for i, data  in enumerate(train_loader):
            input, target_obj_1, target_relation, target_obj_2 = data
            input = torch.transpose(input, 1, 0)
            #print(input.shape)
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
            torch.save(model.state_dict(), "model_after_"+str(epoch))
        if epoch > 0 and epoch % 10000 == 0:
            test()
    # # See what the scores are after training
    # with torch.no_grad():
    #     inputs = prepare_sequence(training_data[0][0], word_to_ix)
    #     tag_scores = model(inputs)

    #     # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    #     # for word i. The predicted tag is the maximum scoring tag.
    #     # Here, we can see the predicted sequence below is 0 1 2 0 1
    #     # since 0 is index of the maximum value of row 1,
    #     # 1 is the index of maximum value of row 2, etc.
    #     # Which is DET NOUN VERB DET NOUN, the correct sequence!
    #     print(tag_scores)

def test():
    model = LSTM()
    model.to(device)
    model.load_state_dict(torch.load("model_after_300"))
    model.eval()
    video_input = getVideoFeaturesFromCsv("../test/test", 'testing')
    model.zero_grad()
    result_obj1 = np.zeros((128,35))
    result_relation = np.zeros((128,82))
    result_obj2 = np.zeros((128,35))
    for i in range(4):
        input = video_input[32*i : 32 * (i + 1)]
        input = torch.transpose(input,1,0)
        probs = model(input)
        result_obj1[32*i : 32 * (i + 1)] = probs[0].detach().to("cpu")
        result_relation[32*i : 32 * (i + 1)] = probs[1].detach().to("cpu")
        result_obj2[32*i : 32 * (i + 1)] = probs[2].detach().to("cpu")
        
    np.savetxt("result_obj1.csv", result_obj1, delimiter=',')
    np.savetxt("result_relation.csv", result_relation, delimiter=',')
    np.savetxt("result_obj2.csv", result_obj2, delimiter=',')
        
#getVideoFeatures("../train/train", 'training')
#getVideoFeatures("../test/test", 'testing')
train()
#test()

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


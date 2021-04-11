import json
import torch
import csv
import numpy as np
from operator import itemgetter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
obj_1_target = torch.zeros(119, device=device, dtype=int)
obj_2_target = torch.zeros(119, device=device, dtype=int)
relation_target = torch.zeros(119, device=device, dtype=int)
file_indices = np.arange(start = 300, stop = 30000, step = 300)
folder_name = 'traintest'
def getTargets():
    json_target = json.load(open('test_annotation.json'))
    idx = 0
    for _, target in json_target.items():
        if idx == 119:
            break
        obj_1_target[idx] = target[0]
        obj_2_target[idx] = target[2]
        relation_target[idx] = target[1]
        idx += 1

def getScore(file_name):
    getTargets()
    score_map = [1, 0.5, 0.33, 0.25, 0.2]
    score = []
    print(file_name)
    with open(file_name) as f:
        obj1 = np.zeros((119,5))
        relation = np.zeros((119,5))
        obj2 = np.zeros((119,5))
        score_obj1 = 0
        score_relation = 0
        score_obj2 = 0
        data = csv.reader(f, delimiter = ' ')

        for row in data:
            row = ','.join(row)
            row = row.split(",")
            if row[0] == 'Id':
                continue
            index_int = int(row[0])
            if index_int % 3 == 0:
                obj1[index_int // 3] = row[1:6]
            elif index_int % 3 == 1:
                relation[index_int // 3] = row[1:6]
            elif index_int % 3 == 2:
                obj2[index_int // 3] = row[1:6]
        for j in range(119):
            for k in range(5):
                # print("obj1: ", obj1[j][k], "target: ", obj_1_target[j])
                # print("relation: ", relation[j][k], "target: ", relation_target[j])
                # print("obj2: ", obj2[j][k], "target: ", obj_2_target[j], "\n")
                if obj1[j][k] == obj_1_target[j]:
                    score_obj1 += score_map[k]
                if relation[j][k] == relation_target[j]:
                    score_relation += score_map[k]
                if obj2[j][k] == obj_2_target[j]:
                    score_obj2 += score_map[k]
            #print(j, score_obj1, score_relation, score_obj2, "\n\n")
        final_score = (score_obj1 + score_relation + score_obj2) / 357.0
        print(final_score)
getScore("48_2700.csv")
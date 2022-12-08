import os
import json 

config_folder_path = 'configs'
dataset_name = 'genea'
suffix = '_typeA'

if suffix == '_typeA' : 
    steps = [ 
        [0, 1, 2, 3, 4, 5, 8, 27, 46, 47, 52, 6, 9, 10, 28, 29, 48, 53, 7, 11, 30, 49, 50, 54, 55, 12, 15, 18, 21, 24, 31, 34, 37, 40, 43, 51, 56, 13, 16, 19, 22, 25, 32, 35, 38, 41, 44, 14, 17, 20, 23, 26, 33, 36, 39, 42, 45]
    ]
elif suffix == '_typeB' : 
    steps = [
        [0, 1, 2, 3, 4, 5, 6], # spine neck
        [8, 27, 46, 47, 52, 9, 10, 28, 29, 48, 53, 7, 11, 30, 49, 50, 54, 55, 12, 15, 18, 21, 24, 31, 34, 37, 40, 43, 51, 56, 13, 16, 19, 22, 25, 32, 35, 38, 41, 44, 14, 17, 20, 23, 26, 33, 36, 39, 42, 45] # remainders
    ]
elif suffix == '_typeC' : 
    steps = [
        [0, 1, 2, 3, 4, 5, 6], # spine neck
        [8, 9, 27, 28, 46, 47, 52, 7], # shoulders pelvis head
        [10, 29, 48, 53, 11, 30, 49, 50, 54, 55, 12, 15, 18, 21, 24, 31, 34, 37, 40, 43, 51, 56, 13, 16, 19, 22, 25, 32, 35, 38, 41, 44, 14, 17, 20, 23, 26, 33, 36, 39, 42, 45] # remainders
    ]
elif suffix == '_typeD' : 
    steps = [
        [0, 1, 2, 3, 4, 5, 6], # spine neck
        [8, 9, 27, 28, 46, 47, 52, 7], # shoulders pelvis head
        [10, 29, 48, 53], # elbows knees 
        [11, 30, 49, 50, 54, 55, 12, 15, 18, 21, 24, 31, 34, 37, 40, 43, 51, 56, 13, 16, 19, 22, 25, 32, 35, 38, 41, 44, 14, 17, 20, 23, 26, 33, 36, 39, 42, 45] # remainders
    ]
elif suffix == '_typeE' : 
    steps = [
        [0, 1, 2, 3, 4, 5, 6], # spine neck
        [8, 9, 27, 28, 46, 47, 52, 7], # shoulders pelvis head
        [10, 29, 48, 53], # elbows knees 
        [11, 30, 49, 54], # wrists ankles
        [50, 55, 12, 15, 18, 21, 24, 31, 34, 37, 40, 43, 51, 56, 13, 16, 19, 22, 25, 32, 35, 38, 41, 44, 14, 17, 20, 23, 26, 33, 36, 39, 42, 45] # remainders
    ]
elif suffix == '_typeF' : 
    steps = [
        [0, 1, 2, 3, 4, 5, 6], # spine neck
        [8, 9, 27, 28, 46, 47, 52, 7], # shoulders pelvis head
        [10, 29, 48, 53], # elbows knees 
        [11, 30, 49, 54], # wrists ankles
        [50, 55, 12, 15, 18, 21, 24, 31, 34, 37, 40, 43], # finger1 heels
        [51, 56, 13, 16, 19, 22, 25, 32, 35, 38, 41, 44, 14, 17, 20, 23, 26, 33, 36, 39, 42, 45] # remainders
    ]
elif suffix == '_typeG' : 
    steps = [
        [0, 1, 2, 3, 4, 5, 6], # spine neck
        [8, 9, 27, 28, 46, 47, 52, 7], # shoulders pelvis head
        [10, 29, 48, 53], # elbows knees 
        [11, 30, 49, 54], # wrists ankles
        [50, 55, 12, 15, 18, 21, 24, 31, 34, 37, 40, 43], # finger1 heels
        [51, 56, 13, 16, 19, 22, 25, 32, 35, 38, 41, 44], # finger2 toes 
        [14, 17, 20, 23, 26, 33, 36, 39, 42, 45] # finger3
    ]
else : 
    raise TypeError(f'check your suffix wording {suffix}')

total_step = len(steps)

idx2jointName = {
    0: 'Hips', 1: 'Spine', 2: 'Spine1', 3: 'Spine2', 4: 'Spine3', 5: 'Neck', 6: 'Neck1', 7: 'Head',
    8: 'RightShoulder', 9: 'RightArm', 10: 'RightForeArm', 11: 'RightHand',
    12: 'RightHandThumb1', 13: 'RightHandThumb2', 14: 'RightHandThumb3',
    15: 'RightHandIndex1', 16: 'RightHandIndex2', 17: 'RightHandIndex3',
    18: 'RightHandMiddle1', 19: 'RightHandMiddle2', 20: 'RightHandMiddle3',
    21: 'RightHandRing1', 22: 'RightHandRing2', 23: 'RightHandRing3',
    24: 'RightHandPinky1', 25: 'RightHandPinky2', 26: 'RightHandPinky3',
    27: 'LeftShoulder', 28: 'LeftArm', 29: 'LeftForeArm', 30: 'LeftHand',
    31: 'LeftHandThumb1', 32: 'LeftHandThumb2', 33: 'LeftHandThumb3',
    34: 'LeftHandIndex1', 35: 'LeftHandIndex2', 36: 'LeftHandIndex3',
    37: 'LeftHandMiddle1', 38: 'LeftHandMiddle2', 39: 'LeftHandMiddle3',
    40: 'LeftHandRing1', 41: 'LeftHandRing2', 42: 'LeftHandRing3',
    43: 'LeftHandPinky1', 44: 'LeftHandPinky2', 45: 'LeftHandPinky3',
    46: 'pCube4',
    47: 'RightUpLeg', 48: 'RightLeg', 49: 'RightFoot', 50: 'RightForeFoot', 51: 'RightToeBase',
    52: 'LeftUpLeg', 53: 'LeftLeg', 54: 'LeftFoot', 55: 'LeftForeFoot', 56: 'LeftToeBase'
}

jointName2idx = {v:k for k, v in idx2jointName.items()}
num_joints = len(idx2jointName)
######### make json #########
config_json = {
    'datasetName' : dataset_name,
    'totalStep' : total_step, 
    'steps' : steps,
    'idx2jointName' : idx2jointName, 
    'jointName2idx' : jointName2idx, 
    'num_joints' : num_joints,
}

with open(f'{os.path.join("..", config_folder_path, dataset_name+suffix+".json")}', 'w') as json_file : 
    json.dump(config_json, json_file, indent=4)
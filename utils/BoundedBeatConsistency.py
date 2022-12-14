import math
import tqdm
import librosa
import numpy as np

def tripoint_angle(p1, p2, p3, radian=False):
    vector_1 = p1 - p2
    vector_2 = p3 - p2
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = round(np.dot(unit_vector_1, unit_vector_2),5)

    angle = np.arccos(dot_product)
    
    if radian :
        return angle
    else:
        return angle*180/math.pi 

def calc_angle(motion_npy, hierarchy):
    body_angle_info = {}

    for frame in range(len(motion_npy)):
        body_angle_info[frame] = {}
        for joint_idx, parent_child_list in hierarchy.items():
            if parent_child_list[1] != 'end' and joint_idx != 0: # ignore root and most end joint
                for child in parent_child_list[1:]:
                    chain = f'{parent_child_list[0]}-{joint_idx}-{child}' # parent - joint idx - children 
                    body_angle_info[frame][chain] = tripoint_angle(motion_npy[frame][parent_child_list[0]], 
                                                                   motion_npy[frame][joint_idx], 
                                                                   motion_npy[frame][child])
    
    return body_angle_info # {frame: {hierarchy_joint: angle}}

def calc_velocity(angle):
    velocity = {}
    for frame in angle.keys():
        velocity[frame]= {}
        if frame != 0: # get velocity from second frame 
            for parent_joint_child in angle[frame].keys():
                velocity[frame][parent_joint_child] = abs(angle[frame][parent_joint_child] - angle[frame -1][parent_joint_child]) * 20 # 20 is fps
                
    return velocity # {frame: {parent_joint_child: angle_velocity(degree/sec)}}

def calc_maac(velocity):
    maac = {pjc : 0 for pjc in velocity[1].keys()} # pjc : parent_joint_child
    total_frame = len(velocity)

    for frame, PJC_vel in velocity.items(): # PJC_vel : {parent_joint_child: angle_velocity(degree/sec)}
        if frame == 0: continue 
        for pjc, vel in PJC_vel.items():
            maac[pjc] += vel

    maac = {pjc : maac[pjc]/total_frame for pjc in maac.keys()}
    return maac #  {parent_joint_child: value}

def calc_average_velocity(velocity):
    average = {frame : 0 for frame in range(len(velocity))}

    for frame in velocity.keys():
        if frame == 0 : continue
        else : 
            average[frame] = sum(velocity[frame].values())/len(velocity[frame])

    return average # {frame: value}

def calc_kinetic_velocity(velocity, maac):
    average = {frame : 0 for frame in range(len(velocity))}

    for frame in velocity.keys():
        if frame == 0 : continue
        else : 
            for pjc, vel in velocity[frame].items():
                average[frame] += vel/maac[pjc]
            average[frame] = average[frame]/len(velocity[frame])
    
    return average # {frame: value} 

def find_motion_local_min(gt_average_velocity, average_velocity, kinetic_velocity, gt_multiple, threshold=1):
    local_min_frame_kinVel = {}
    result = {}
    
    for frame in kinetic_velocity.keys():
        if frame == 0 or frame == len(kinetic_velocity)-1:
            continue # ignore first and last frame
        
        is_local_min = kinetic_velocity[frame-1] - kinetic_velocity[frame] > 0 and \
            kinetic_velocity[frame] - kinetic_velocity[frame+1] < 0 # local min condition
        is_over_threshold = kinetic_velocity[frame -1] > threshold or kinetic_velocity[frame+1] > threshold

        if is_local_min and is_over_threshold:
            local_min_frame_kinVel[frame] = kinetic_velocity[frame]
    
    average_threshold = (sum(gt_average_velocity.values())/len(gt_average_velocity)) * gt_multiple
    
    for frame in local_min_frame_kinVel.keys(): # based on local min frame, find motion peak
        if average_velocity[frame - 1] > average_threshold or average_velocity[frame+1] > average_threshold:
            result[frame] = kinetic_velocity[frame]

    return result # {frame: value} 

def make_onset_peaks(audio):
    if isinstance(audio, str): 
        y, sr = librosa.load(audio)
    else : 
        y, sr = audio
    onset_env = librosa.onset.onset_strength_multi(y = y, sr = sr, n_fft=sr//20, hop_length=sr//20)
    onset_env = onset_env[0]
    detect = librosa.onset.onset_detect(y, sr, onset_envelope=onset_env, units='frames', hop_length= sr//20)

    peak = {frame: onset_env[frame] for frame in detect}
    average = sum(peak.values())/len(peak)

    for key in peak:
        peak[key] = peak[key]/average
    
    return peak # {frame: value} 

def calc_BC1(audio_peak, motion_peak, sigma, quiet=False, mute=1):
    nearist = {}
    near_diff = {}
    mute = int(mute*20)
    mute_check = 0
    sigma = sigma
    motion_peak_keys = list(motion_peak.keys()) # motion
    audio_peak_keys = list(audio_peak.keys()) # audio

    
    progress = audio_peak_keys if quiet else tqdm.tqdm(audio_peak_keys) 
    for i in progress :
        if mute_check <= i:
            for hierarchy in range(1,12):
                if (i + hierarchy) in motion_peak_keys:
                    nearist[i] = i+hierarchy
                    mute_check = i+mute
                    break
                
                elif (i - hierarchy) in motion_peak_keys:
                    nearist[i] = i-hierarchy
                    mute_check = i+mute
                    break
                
                elif hierarchy == 11:
                    nearist[i] = 'check'
                    break

    for audio, motion in nearist.items():
        if motion == 'check':
            pass
        else: 
            near_diff[audio] = math.exp(-(abs(audio_peak[audio] - motion_peak[motion])**2)/(2*(sigma**2)))
            
    if len(near_diff) != 0:
        BC_result = sum(near_diff.values())/len(near_diff)
        
    else: BC_result = 0
    
    return BC_result

def evaluate_BC(gt_motion_numpy_path, motion_numpy_path, audio_file_path, gt_multiple, local_min_threshold, bc_sigma, mute, quiet=False):
    JointIndex_with_ParentChildList = { # Trinity Version
        0: ['root', 1, 47,52], 
        1: [0, 2], 
        2: [1, 3], 
        3: [2, 4], 
        4: [3, 5, 8, 27, 46], 
        5: [4, 6], 
        6: [5, 7], 
        7: [6, 'end'], 
        8: [4, 9], 
        9: [8, 10], 
        10: [9, 11], 
        11: [10, 12, 15, 18, 21, 24], 
        12: [11, 13], 
        13: [12, 14], 
        14: [13, 'end'], 
        15: [11, 16], 
        16: [15, 17], 
        17: [16, 'end'], 
        18: [11, 19], 
        19: [18, 20], 
        20: [19, 'end'], 
        21: [11, 22], 
        22: [21, 23], 
        23: [22, 'end'], 
        24: [11, 25], 
        25: [24, 26], 
        26: [25, 'end'], 
        27: [4, 28], 
        28: [27, 29], 
        29: [28, 30], 
        30: [29, 31, 34, 37, 40, 43], 
        31: [30, 32], 
        32: [31, 33], 
        33: [32, 'end'], 
        34: [30, 35], 
        35: [34, 36], 
        36: [35, 'end'], 
        37: [30, 38], 
        38: [37, 39], 
        39: [38, 'end'], 
        40: [30, 41], 
        41: [40, 42], 
        42: [41, 'end'], 
        43: [30, 44], 
        44: [43, 45], 
        45: [44, 'end'], 
        46: [4, 'end'], 
        47: [0, 48], 
        48: [47, 49], 
        49: [48, 50], 
        50: [49, 51], 
        51: [50, 'end'], 
        52: [0, 53], 
        53: [52, 54], 
        54: [53, 55], 
        55: [54, 56], 
        56: [55, 'end']
        }
    
    spine_head_list = [0, 1, 2, 3, 4, 5, 6]
    arm_leg_list = [8, 9, 10, 27, 28, 29, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56]
    hand_list = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
    
    spine_head = {idx: JointIndex_with_ParentChildList[idx] for idx in spine_head_list}
    arm_leg = {idx: JointIndex_with_ParentChildList[idx] for idx in arm_leg_list}
    hand = {idx: JointIndex_with_ParentChildList[idx] for idx in hand_list}
    
    body_parts = {'vetrebral' : spine_head, 
                  'limb' : arm_leg, 
                  'hand' : hand}

    results = {}
    
    for name, part_hierarchy in body_parts.items():
        if isinstance(gt_motion_numpy_path, str) : 
            gt_motion_numpy = np.load(gt_motion_numpy_path)
        else : 
            gt_motion_numpy = gt_motion_numpy_path
        
        if isinstance(motion_numpy_path, str) : 
            motion_numpy = np.load(motion_numpy_path)
        else : 
            motion_numpy = motion_numpy_path

        gt_angle = calc_angle(gt_motion_numpy, part_hierarchy)
        gt_velocity = calc_velocity(gt_angle)
        gt_maac = calc_maac(gt_velocity)
        gt_average_velocity = calc_average_velocity(gt_velocity)


        angle = calc_angle(motion_numpy, part_hierarchy)
        velocity = calc_velocity(angle)
        maac = calc_maac(velocity)
        average_velocity = calc_average_velocity(velocity)

        kin_vel = calc_kinetic_velocity(velocity, maac)
        motion_feat = find_motion_local_min(gt_average_velocity, average_velocity, kin_vel, gt_multiple, threshold=local_min_threshold)
        audio_feat = make_onset_peaks(audio_file_path)
        bc = calc_BC1(audio_feat, motion_feat, sigma = bc_sigma, quiet=quiet, mute=mute)
        
        results[name] = bc
        
    return results


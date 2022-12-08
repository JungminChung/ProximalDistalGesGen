import os
import cv2
import numpy as np 
import pyquaternion as pyq
from tqdm import tqdm

def read_bvh(bvh_path): 
    with open(bvh_path, 'r') as bvh:
        bvh_content = bvh.readlines()
    return bvh_content

def split_bvh_content(bvh_content):
    for idx, line in enumerate(bvh_content) : 
        if line.startswith('MOTION'):
            motion_idx = idx 
            break 
    hierarchy = bvh_content[: motion_idx]
    bvh_info  = bvh_content[motion_idx: motion_idx+3]
    bvh_frame = bvh_content[motion_idx+3: ]

    return hierarchy, bvh_info, bvh_frame

def create_hierarchy_nodes(hierarchy, depth_space=2, remove_endsite=True):
    joint_offsets = []
    joint_names = []
    joint_level = []

    for line in hierarchy:
        line = line.replace('\t', ' ' * depth_space)
        level = int((len(line) - len(line.lstrip())) / depth_space)

        line = line.split()
        if not len(line) == 0:
            line_type = line[0]
            if line_type == 'OFFSET':
                offset = np.array([float(line[1]), float(line[2]), float(line[3])])
                joint_offsets.append(offset)
            elif line_type == 'ROOT' or line_type == 'JOINT':
                joint_names.append(line[1])
                joint_level.append(level)
            elif line_type == 'End':
                joint_names.append('End Site')
                joint_level.append(level)
    
    ## remove end site 
    if remove_endsite : 
        removed_joint_offsets = []
        removed_joint_names = []
        removed_joint_level = []
        
        for offset, name, level in zip(joint_offsets, joint_names, joint_level): 
            if not name == 'End Site' : 
                removed_joint_offsets.append(offset)
                removed_joint_names.append(name)
                removed_joint_level.append(level)

        joint_offsets = removed_joint_offsets
        joint_names = removed_joint_names
        joint_level = removed_joint_level

    nodes = []
    for idx, (offset, name, level) in enumerate(zip(joint_offsets, joint_names, joint_level)): 
        cur_level = level
        parent = None
        children = []
        for back_idx in range(idx+1, len(joint_names)):
            if joint_level[back_idx] == cur_level : 
                break 
            if joint_level[back_idx] == cur_level + 1 : 
                children.append(back_idx)
        for front_idx in range(idx-1, -1, -1):
            if joint_level[front_idx] == cur_level - 1 :
                parent = int(front_idx)
                break 

        node = dict([('idx', idx),
                     ('name', name), 
                     ('parent', parent), 
                     ('children', children), 
                     ('offset', joint_offsets[idx]), 
                     ('rel_degs', None), 
                     ('abs_qt', None), 
                     ('rel_pos', None), 
                     ('abs_pos', None)])

        if idx == 0:
            node['rel_pos'] = node['abs_pos'] = [float(0), float(60), float(0)]
            node['abs_qt'] = pyq.Quaternion()
        nodes.append(node)

    return nodes

def rearrange_joint_1dto2d(joint_1d):
    joints = [] 
    coordinate = []
    for idx, coo in enumerate(joint_1d): 
        coordinate.append(float(coo))
        if (idx+1) % 3 == 0 : 
            joints.append(coordinate)
            coordinate = []
    return joints

def rearrange_joint_2dto1d(joint_2d, target_joint, num_total_joint):
    results = []
    focus_joint = 0 
    for joint_idx in range(num_total_joint): 
        if joint_idx in target_joint : 
            results.extend([joint_2d[focus_joint][0], joint_2d[focus_joint][1], joint_2d[focus_joint][2]])
            focus_joint += 1 
        else : 
            results.extend([0.0, 0.0, 0.0])
    return results

def rot_vec_to_abs_pos_vec(line, nodes):
    assert (len(line) // 3) * 3 == len(line), 'check the num of rotation point and joint num'

    node_idx = 0 
    for i in range(len(line)//3): # 전체 joint 갯수
        stepi = i*3
        z_deg = float(line[stepi])
        x_deg = float(line[stepi+1])
        y_deg = float(line[stepi+2])

        if nodes[node_idx]['name'] == 'End Site':
            node_idx = node_idx + 1
        nodes[node_idx]['rel_degs'] = [z_deg, x_deg, y_deg]
        current_node = nodes[node_idx]

        node_idx = node_idx + 1

    for start_node in nodes:
        abs_pos = np.array([0, 60, 0])
        current_node = start_node
        if start_node['children'] is not None: 
            for child_idx in start_node['children']:
                child_node = nodes[child_idx]

                child_offset = np.array(child_node['offset'])
                qz = pyq.Quaternion(axis=[0, 0, 1], degrees=start_node['rel_degs'][0])
                qx = pyq.Quaternion(axis=[1, 0, 0], degrees=start_node['rel_degs'][1])
                qy = pyq.Quaternion(axis=[0, 1, 0], degrees=start_node['rel_degs'][2])
                qrot = qz * qx * qy
                offset_rotated = qrot.rotate(child_offset)
                child_node['rel_pos']= start_node['abs_qt'].rotate(offset_rotated)

                child_node['abs_qt'] = start_node['abs_qt'] * qrot

        while current_node['parent'] is not None:

            abs_pos = abs_pos + current_node['rel_pos']
            current_node = nodes[current_node['parent']]
        start_node['abs_pos'] = abs_pos

    line = []
    for node in nodes:
        line.append(node['abs_pos'])

    return np.array(line)

def nodes_to_abs_vector(nodes): 
    return [node['abs_pos'] for node in nodes]

def nodes_to_child_list(nodes): 
    return [node['children'] for node in nodes]

def make_image_from_abs_childlist(abs_vector, child_list, dataset='trinity'): 
    bg_size = 512 
    size_mag = 2.0 if dataset == 'trinity' else 300.0

    x_offset = bg_size * (1/2)
    y_offset = bg_size * (1/3) if dataset == 'trinity' else bg_size * (2/3)

    img = np.zeros([bg_size, bg_size, 3],dtype=np.uint8)
    img.fill(255)
    
    for vec, children in zip(abs_vector, child_list) : 
        if any([(v is None) or (np.isnan(v)) for v in vec]): continue 
        x = int((vec[0] * size_mag) + x_offset)
        y = int(bg_size - ((vec[1] * size_mag) + y_offset)) if dataset == 'trinity' else int(((vec[1] * size_mag) + y_offset))

        color = (255, 0, 0)
        img = cv2.circle(img, (x, y), 3, color, -1)
        
        for child_idx in children :
            if any([(v is None) or (np.isnan(v)) for v in abs_vector[child_idx]]): continue 
            target_x = int(abs_vector[child_idx][0] * size_mag + x_offset)
            target_y = int(bg_size - ((abs_vector[child_idx][1] * size_mag) + y_offset)) if dataset == 'trinity' else int(((abs_vector[child_idx][1] * size_mag) + y_offset))
            img = cv2.line(img, (x, y), (target_x, target_y), (0, 0, 180), 1)
                
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def make_image_from_nodes(nodes):
    abs_vector = nodes_to_abs_vector(nodes) 
    child_list = nodes_to_child_list(nodes)
    
    img = make_image_from_abs_childlist(abs_vector, child_list)
    return img

def get_idx2name_from_nodes(nodes):
    return {i:n['name'] for i, n in enumerate(nodes)}

def reverse_dictionary(dict):
    return {v:k for k, v in dict.items()}

def get_bvh_vector(bvh_path, depth_space=2):
    with open(bvh_path, 'r') as bvh : 
        bvh_content = bvh.readlines()
    hierarchy, bvh_info, bvh_frame = split_bvh_content(bvh_content)
    total_time = int(bvh_info[1].split()[-1]) * float(bvh_info[2].split()[-1])
    print(f'BVH {os.path.split(bvh_path)[-1]} has {int(total_time//60)} min {total_time%60:.2f} sec movement length.')
    
    fps = round(1/float(bvh_info[2].split()[-1]))

    nodes = create_hierarchy_nodes(hierarchy, depth_space=depth_space)

    results_bvh_frame = [[float(x) for i, x in enumerate(line.split()) if i > 2] for line in bvh_frame]
    if fps == 100 : # 100 FPS -> 20 FPS (every 5th line)
        results_bvh_frame = results_bvh_frame[::5]
    elif fps == 60 : # 60 FPS -> 20 FPS (every 3rd line)
        results_bvh_frame = results_bvh_frame[::3]
    elif fps == 24 : # 24 FPS -> 20 FPS (del every 6th line)
        del results_bvh_frame[::6]
    elif fps == 30 : 
        del results_bvh_frame[::3]
    
    output_vectors = [] 
    progress = tqdm(results_bvh_frame, ncols=100)
    for frame in progress : 
        progress.set_description(f'Loading bvhs... \t ')
        output_vectors.append(rot_vec_to_abs_pos_vec(frame, nodes))
    
    return np.array(output_vectors)

def align_rot_vec(line):
    assert (len(line) // 3) * 3 == len(line), 'check the num of rotation point and joint num'

    results = []
    for i in range(len(line)//3): # 전체 joint 갯수
        stepi = i*3
        z_deg = float(line[stepi])
        x_deg = float(line[stepi+1])
        y_deg = float(line[stepi+2])

        results.append(np.array([z_deg, x_deg, y_deg]))

    return np.array(results)

def save_hierarchy(bvh_path, assist_npy_folder, name='hierarchy'): 
    bvh_content = read_bvh(bvh_path)
    hierarchy, _, _ = split_bvh_content(bvh_content)
    with open(os.path.join(assist_npy_folder, name+'.bvh'), 'w') as h : 
        for line in hierarchy :
            h.write(line)
    nodes = create_hierarchy_nodes(hierarchy)
    root_offset = np.array([0, 60, 0])
    offset_list = [
        root_offset # ROOT offest and to make it [0, 60, 0]
    ] 

    for idx in range(1, len(nodes)) : 
        parent_idx = nodes[idx]['parent']
        offset_list.append(
            offset_list[parent_idx] + nodes[idx]['offset'] 
        )

    one_bvh_npy = np.array(offset_list).flatten()
    one_bvh_npy = np.expand_dims(one_bvh_npy, 0)
    np.save(os.path.join(assist_npy_folder, name+'.npy'), one_bvh_npy)

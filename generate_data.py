import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torchvision
import pickle
import argparse

def GetRandomTrajectory(length, canvas_size, batch_size):

    # Initial position uniform random inside the box.
    y = np.random.rand(batch_size)
    x = np.random.rand(batch_size)

    # Choose a random velocity.
    theta = np.random.rand(batch_size) * 2 * np.pi
    v_y = np.sin(theta)
    v_x = np.cos(theta)

    start_y = np.zeros((length, batch_size))
    start_x = np.zeros((length, batch_size))
    for i in range(length):
        # Take a step along velocity.
        #y += v_y * 0.01
        #x += v_x * 0.01
        ### change the step randomly 
        step = np.random.uniform(low=0.01, high=0.05)
        y += v_y * step
        x += v_x * step
        

        # Bounce off edges.
        for j in range(batch_size):
            if x[j] <= 0:
                x[j] = 0
                v_x[j] = -v_x[j]
            if x[j] >= 1.0:
                x[j] = 1.0
                v_x[j] = -v_x[j]
            if y[j] <= 0:
                y[j] = 0
                v_y[j] = -v_y[j]
            if y[j] >= 1.0:
                y[j] = 1.0
                v_y[j] = -v_y[j]
        start_y[i, :] = y
        start_x[i, :] = x

    # Scale to the size of the canvas.
    start_y = (canvas_size * start_y).astype(np.int32)
    start_x = (canvas_size * start_x).astype(np.int32)
    return start_y, start_x

def Overlap(a, b):
    """ Put b on top of a."""
    overlap_map = (np.array(a) != 0) & (np.array(b) != 0)  ### if there are True, there is event (overlap)
    return np.maximum(a, b), (len(np.where(overlap_map)[0]) != 0)
    #return b

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='The directory of MNIST', type=str, default='./data_dir/')
    parser.add_argument('--generate_data_path', help='The directory for saving generate data', type=str, default='./generate_data/')
    args = parser.parse_args()

    mnist_dataset = torchvision.datasets.MNIST(args.data_path, train=True, download=True)
    mnist_data_num = len(mnist_dataset)
    mnist_indices = np.arange(mnist_data_num)
    np.random.shuffle(mnist_indices)
    row = 0

    length = 100
    image_size = 128
    digit_size = 28
    total_nums = 20000

    start_y, start_x = GetRandomTrajectory(length, image_size - digit_size, total_nums * 2)
    generate_data = np.zeros((total_nums, length, image_size, image_size), dtype=np.float32)
    generate_event = np.zeros((total_nums, length))
    for j in tqdm(range(total_nums)):
        for n in range(2):
           ind = mnist_indices[row]
           row += 1
           if row == mnist_data_num:
               row = 0
               np.random.shuffle(mnist_indices)
           digit_image, digit_label = mnist_dataset[ind]
           for i in range(length):
               top    = start_y[i, j * 2 + n]
               left   = start_x[i, j * 2 + n]
               bottom = top  + digit_size ### one image size is 28 x 28
               right  = left + digit_size ### one image size 28 x 28
               generate_data[j, i, top:bottom, left:right], generate_event[j, i] = Overlap(generate_data[j, i, top:bottom, left:right], digit_image)
               generate_data[j, i] += np.random.normal(0,10,(image_size, image_size)) ### adding noise

    ### cut sequence and simulate censoring
    cut_idxs = []
    for label in generate_event:
	label_diff = np.diff(label)
	if 1 in label_diff:
	   cut_idxs.append(np.where(label_diff == 1)[0][0] + 1)
        else:
           cut_idxs.append(len(label))
    
    censor_indicator = np.random.choice(2, len(cut_idxs), p=[0.85, 0.15])
    final_len = []
    for i, item in enumerate(cut_idxs):
	if censor_indicator[i]:
           if item > 1:
              final_len.append(np.random.randint(1, item, 1)[0])
           else:
              final_len.append(item)
        else:
           final_len.append(item)
    
    all_idxs = np.arange(len(final_len))
    train_idxs = np.random.choice(all_idxs, int(len(all_idxs) * 0.8), replace=False)
    val_idxs = np.random.choice(list(set(all_idxs) - set(train_idxs)), int(len(all_idxs) * 0.1), replace=False)
    test_idxs = np.array(list(set(all_idxs) - set(train_idxs).union(set(val_idxs)))) 
    
    ### increase the sequence complexity
    resize_size = 32
    mnist_transform = torchvision.transforms.Compose([
    	torchvision.transforms.ToTensor(), 
    	torchvision.transforms.Resize(resize_size), ### resize the image into 32 x 32
    	torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    norm_train_dataset1, norm_train_dataset2 = [], []
    train_labels = []
    for idx in tqdm(train_idx):
       	norm_data = [mnist_transform(item) for item in generate_data[idx][:final_len[idx]]]
       	norm_train_dataset1.append([item.squeeze()[:resize_size//2,:].flatten() for item in norm_data])
       	norm_train_dataset2.append([item.squeeze()[resize_size//2:,:].flatten() for item in norm_data])
    
    norm_val_dataset1, norm_val_dataset2 = [], []
    val_labels = []
    for idx in tqdm(val_idx):
    	norm_data = [mnist_transform(item) for item in generate_data[idx][:final_len[idx]]]
    	norm_val_dataset1.append([item.squeeze()[:resize_size//2,:].flatten() for item in norm_data])
    	norm_val_dataset2.append([item.squeeze()[resize_size//2:,:].flatten() for item in norm_data])

    norm_test_dataset1, norm_test_dataset2 = [], []
    test_labels = []
    for idx in tqdm(test_idx):
    	norm_data = [mnist_transform(item) for item in generate_data[idx][:final_len[idx]]]
    	norm_test_dataset1.append([item.squeeze()[:resize_size//2,:].flatten() for item in norm_data])
    	norm_test_dataset2.append([item.squeeze()[resize_size//2:,:].flatten() for item in norm_data])
    
    train_timestamps = [np.arange(len(item)) for item in train_labels]
    val_timestamps = [np.arange(len(item)) for item in val_labels]
    test_timestamps = [np.arange(len(item)) for item in test_labels] 
    
    with open(args.generate_data_path + 'generate_data.pkl', 'wb') as f:
    pickle.dump({
        'train': {'data1': norm_train_dataset1, 'data2': norm_train_dataset2, 'label': train_labels, 'timestamp': train_timestamps, 'id': train_idx},
        'val': {'data1': norm_val_dataset1, 'data2': norm_val_dataset2, 'label':val_labels, 'timestamp': val_timestamps, 'id': val_idx},
        'test': {'data1': norm_test_dataset1, 'data2': norm_test_dataset2, 'label': test_labels, 'timestamp': test_timestamps, 'id': test_idx},
    }, f, protocol=pickle.HIGHEST_PROTOCOL)
     
    

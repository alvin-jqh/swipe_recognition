from torch.utils.data import Dataset
import torch
import glob
import os
import pandas as pd
import numpy as np
import re

vocabulary = {'_': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8,
              'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16,
              'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24,
              'y': 25, 'z': 26}
class SwipeDataset(Dataset):
    # dataset containing the words, swipe touchpoints and tokenized letters
    def __init__(self, data_dir, batch = True, batch_first = True):
        # the directory where the data is being stored
        self.data_dir = data_dir
        # controls whether to make the tensors batchable, adds extra dimension to data
        self.batch = batch
        self.batch_first = batch_first
        # the inputs for the model, in the form (x, y, t, v, a, angle)
        # the first coordinate is shifted to (0,0) at time 0
        self.data = []
        # the words being swipes
        self.words = []
        # words represented by index
        self.word_tensors = []

        # list of all the ndjson files
        data_files = glob.glob(os.path.join(data_dir, "*.ndjson"))
        for filename in data_files:
            df = pd.read_json(filename, lines=True, encoding="utf-16")
            # remove any datapoints where there are less than 5 datapoints
            df = df[df["swipe"].apply(lambda x: isinstance(x, list) and 5 < len(x) < 350)].reset_index(drop=True)
            # drop any data points where there are special characters
            df = df[df["word"].apply(lambda x: re.match(r'^[a-z]+$', x) is not None)].reset_index(drop=True)
            # Filter out swipes that result in division by zero during speed calculation
            df = df[df["swipe"].apply(self.check_division_by_zero)].reset_index(drop=True)
            words = df["word"].tolist()
            inputs = df["swipe"].apply(self.handle_data).tolist()
            encoded_words = df["word"].apply(self.encode_word).tolist()
            for input, word, encoded_word in zip(inputs, words, encoded_words):
                self.data.append(input)
                self.words.append(word)
                self.word_tensors.append(encoded_word)                 

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        word = self.words[index]
        word_tensor = self.word_tensors[index]

        return data, word, word_tensor
    
    def check_division_by_zero(self, swipes):
        swipes = np.array(swipes)
        time = swipes[:, 2]
        dt = np.diff(time)
        return not np.any(dt == 0) # return True if there are no zero values in dt
    
    def handle_data(self, swipes):
        swipes = np.array(swipes)
        # get the maximum value for x and y
        max_x = np.max(swipes[:,0])
        max_y = np.max(swipes[:,1])

        # shift so the first point is always (0,0,0)
        shifted_points = swipes - swipes[0]
        shifted_x = shifted_points[:, 0]
        shifted_y = shifted_points[:, 1]

        # divide by max
        normalized_x = shifted_x / max_x
        normalized_y = shifted_y / max_y
        time = shifted_points[:, 2]

        dx = np.diff(normalized_x)
        dy = np.diff(normalized_y)
        dt = np.diff(time)    

        distance = np.sqrt(dx**2 + dy**2)
        speed = distance / dt  # in units/ms
        padded_speed = np.concatenate(([speed[0]], speed))

        dv = np.diff(padded_speed)
        acceleration = dv / dt  # in units/ms
        padded_acceleration = np.concatenate(([acceleration[0]], acceleration))

        angles = np.arctan2(dy, dx)  # in radians
        padded_angles = np.concatenate(([angles[0]], angles))

        inputs = np.column_stack((normalized_x,             # x shifted to 0    
                                  normalized_y,             # y shifted to 0
                                  time,                     # time shifted to 0
                                  padded_speed,             # speed in pixels/ms
                                  padded_acceleration,      # accel in pixels/ms/ms
                                  padded_angles,            # in radians between -pi and pi
                                  )).tolist()
        if self.batch:
            if self.batch_first:
                # returns a tensor in shape (1, T, 6), T is length of sequence, allows for batching
                return torch.Tensor(inputs).unsqueeze(0)
            else:
                # returns a tensor in shape (T, 1, 6), T is length of sequence, allows for batching
                return torch.Tensor(inputs).unsqueeze(1)
        else:  
            # returns a tensor in shape (T, 6), T is length of sequence
            return torch.Tensor(inputs)
        
    def encode_word(self, word):
        encoded = []
        for letter in word:
            encoded.append(vocabulary[letter])
        
        if self.batch:
            # returns a (1, S) where S is the length of the word
            return torch.Tensor(encoded).unsqueeze(0).long()
        else:
            # returns list with length of word
            return torch.Tensor(encoded).long()
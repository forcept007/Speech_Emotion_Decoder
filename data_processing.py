import os
import pickle
import numpy as np

def trim_array(arr):
    if np.all(arr == 0):  # Check if the array is all zeros
        return arr
    first_non_zero = np.argmax(arr != 0)
    last_non_zero = len(arr) - np.argmax(arr[::-1] != 0)
    return arr[first_non_zero:last_non_zero]

def pad_trunc_audio(audio_data, target_length):
    standardized_data = []
    for data in audio_data:
        if len(data) < target_length:
            padded_data = np.pad(data, (0, target_length - len(data)), 'constant', constant_values=(0, 0))
            standardized_data.append(padded_data)
        elif len(data) > target_length:
            truncated_data = data[:target_length]
            standardized_data.append(truncated_data)
        else:
            standardized_data.append(data)
    return standardized_data

def load_audio_data(directory="train"):
    allaudios = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pkl"):
                with open(os.path.join(root, file), 'rb') as f:
                    loadedaudios = pickle.load(f)
                    allaudios.append(loadedaudios)
    return allaudios
import torch
import pickle
from data_processing import trim_array, pad_trunc_audio
import os

def process_file(file_path, model_best, target_length):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    #apply preprocessing
    audio_data = data["audio_data"]
    audio_trimmed = trim_array(audio_data)
    
    audio_pad_trunced = pad_trunc_audio([audio_trimmed], target_length)
    audio_data_tensor = torch.tensor(audio_pad_trunced).unsqueeze(0)

    # print(f"Original shape: {audio_data.shape}")
    # print(f"Trimmed shape: {audio_trimmed.shape}")
    # print(f"Pad/Truncated shape: {len(audio_pad_trunced[0])}")
    # print(f"Tensor shape: {audio_data_tensor.shape}")


    valence = model_best(audio_data_tensor).item()
    return os.path.basename(file_path), valence
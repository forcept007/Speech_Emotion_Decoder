import os
import math
import random
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from train import train_model, loss_plot
from torch.utils.data import Dataset, DataLoader

from inference import process_file
from model import CNN1d, Normalization
from data_processing import load_audio_data, trim_array, pad_trunc_audio
from hyperparameter_tuning import generate_sequences, hyper_train_setup, plot_search


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('imports done')

# Load Data
allaudios = load_audio_data("train")

audio_data = []
valence = []
audio_lengths = []
# i=0
for audio in allaudios:
    # Get Rid of front and end trailing zeros
    audio_trimmed = trim_array(audio['audio_data'])
    audio_data.append(audio_trimmed)

    audio_length = len(audio_trimmed)
    audio_lengths.append(audio_length)
    valence.append(audio['valence'])
    # i+=1
    # if i==100:
    #     break


mean_length = np.mean(audio_lengths)
median_length = np.median(audio_lengths)
std_dev = np.std(audio_lengths)
min_length = np.min(audio_lengths)
max_length = np.max(audio_lengths)


standardized_audios = pad_trunc_audio(audio_data, target_length=int(np.percentile(audio_lengths, 95)))
print("Done padding")


class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

size_train = int(round(len(standardized_audios) * 0.8))
X_train = standardized_audios[:size_train]
X_test = standardized_audios[size_train:]
y_train = valence[:size_train]
y_test = valence[size_train:]

flatten = np.concatenate(X_train)
mean = np.mean(flatten)
std = flatten.std()
normalization = Normalization(mean, std)

batch_size = 64

train_dataset = AudioDataset(X_train, y_train)
test_dataset = AudioDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,) # you can speed up the host to device transfer by enabling pin_memory.
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,) # you can speed up the host to device transfer by enabling pin_memory.

print("Done with data loaders")

#################################################################################################### Find Best Optimizer
names = [] # Initialize an empty list names to store optimizer names for visualization of results.
learning_rate = 0.001
num_epochs = 15

for opt in [optim.SGD, optim.Adagrad, optim.Adam]:
    names.append(opt.__name__)
    model_1d = CNN1d([], [16, 32, 64, 128, 256], nn.ReLU, normalization).to(device)

    if opt is optim.SGD:
        optimizer = opt(model_1d.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    else:
        optimizer = opt(model_1d.parameters(), lr=learning_rate)

    train_loss_lst, val_loss_lst = train_model(model_1d, optimizer, train_dataloader, test_dataloader, num_epochs)
    plt.plot(val_loss_lst)
    # Print the validation accuracy and loss in the last epoch
    print(f'{opt.__name__}\n\tValidation Loss: {val_loss_lst[-1]:.5}\n')
    

#Display a legend in the plot for optimizer names.
plt.legend(names)
#Label the x-axis as "Epoch" and the y-axis as "Validation Accuracy."
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
#Show the plot containing the accuracy evolution over epochs for each optimizer.
plt.savefig("optimizers_comparison_new.png")
plt.show()


#################################################################################################### Coarse Random Search
coarse_trials = 30
num_epochs = 13
coarse_results = []

hidden_sizes_list = generate_sequences(16,128)
hidden_options = len(hidden_sizes_list)


for i in range(coarse_trials):
    lr = 10**random.uniform(math.log10(0.001), math.log10(0.1))
    index = int(2**random.uniform(math.log2(1), math.log2(hidden_options-1)))
    hidden_sizes = hidden_sizes_list[index]
    val_loss = hyper_train_setup(hidden_sizes, lr, num_epochs, train_dataloader, test_dataloader, normalization)

    coarse_results.append({'lr': lr, 'index':index, 'hidden_size': hidden_sizes, 'loss': val_loss})

    print(f"{i+1}. Learning rate: {lr:.4} and hidden sizes: {hidden_sizes}")
    print(f"\tValidation loss: {val_loss:.5}\n")

# Find the best parameters from coarse search
best_coarse_params = min(coarse_results, key=lambda x: x['loss'])
print(f"Best parameters found:\n - Learning rate: {best_coarse_params['lr']:.5}\n - Hidden sizes: {best_coarse_params['hidden_size']}\n - Validation loss: {best_coarse_params['loss']:.5}%")


fine_trials = 10
fine_results = []
hidden_options = len(hidden_sizes_list)


for _ in range(fine_trials):
    lr = 2**random.uniform(np.log2(0.5 * best_coarse_params['lr']), np.log2(1.5 * best_coarse_params['lr']))

    index = float('inf')
    while index > hidden_options-1:
        index = random.randint(int(0.8 * best_coarse_params['index']), int(1.2 * best_coarse_params['index']) + 1) # not inclusive on the end
    hidden_sizes = hidden_sizes_list[index]
    val_loss = hyper_train_setup(hidden_sizes, lr, num_epochs, train_dataloader, test_dataloader, normalization)

    fine_results.append({'lr': lr, 'index':index, 'hidden_size': hidden_sizes, 'loss': val_loss})

    print(f"Learning rate: {lr:.4} and hidden sizes: {hidden_sizes}")
    print(f"\tValidation loss: {val_loss:.5}\n")

# Find the best parameters from fine search
best_fine_params = min(fine_results, key=lambda x: x['loss'])

print(f"Best parameters found with coarse search:\n - Learning rate: {best_coarse_params['lr']:.5}\n - Hidden sizes: {best_coarse_params['hidden_size']}\n - Validation loss: {best_coarse_params['loss']:.5}%")
print(f"Best parameters found with fine search:\n - Learning rate: {best_fine_params['lr']:.5}\n - Hidden sizes: {best_fine_params['hidden_size']}\n - Validation loss: {best_fine_params['loss']:.5}%")
plot_search(coarse_results + fine_results, "lr", "index", 'loss')

#################################################################################################### Run and Save Best model
model_best = CNN1d([], best_coarse_params['hidden_size'], nn.ReLU, normalization).to(device)
optimizer = optim.Adagrad(model_best.parameters(), lr=best_coarse_params['lr'])
train_loss_lst, val_loss_lst = train_model(model_best, optimizer, train_dataloader, test_dataloader, 13)
loss_plot(train_loss_lst, val_loss_lst, f"best_model_hidden_size_{hidden_sizes}_lr_{learning_rate:.4}_epochs_{num_epochs}.png")

save_path = f"best_coarse_model_adagrad_{num_epochs}_epochs_with_normalization_new"

torch.save(model_best, save_path)


############################################################################# Testing Data
# List to store results
results = []

# Iterate through all files in the folder
for filename in os.listdir("test"):
    if filename.endswith('.pkl'):
        file_path = os.path.join("test", filename)
        file_id, valence = process_file(file_path, model_best, target_length=int(np.percentile(audio_lengths, 95)))
        results.append((file_id, valence))

# Create a DataFrame for better visualization and potential saving to CSV
results_df = pd.DataFrame(results, columns=['ID', 'valence'])


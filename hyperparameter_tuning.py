import matplotlib.pyplot as plt
from model import CNN1d
import torch.nn as nn
from train import train_model, loss_plot
import torch.optim as optim
import torch

def generate_sequences(start=16, max_value=512):
    sequences = [[start]] 
    finished_sequences = []

    while sequences:
        new_sequences = []
        for seq in sequences:
            last_value = seq[-1]
            
            if seq.count(last_value) < 2:
                new_seq = seq + [last_value]
                new_sequences.append(new_seq)
            
            doubled_value = last_value * 2
            if doubled_value <= max_value:
                new_seq = seq + [doubled_value]
                new_sequences.append(new_seq)
        
        finished_sequences.extend(sequences)
        sequences = new_sequences

    return finished_sequences

def hyper_train_setup(hidden_sizes, learning_rate, num_epochs, train_dataloader, test_dataloader, normalization):
    # from main import normalization, device, train_dataloader, test_dataloader
    #  # Create the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_1d = CNN1d([], hidden_sizes, nn.ReLU, normalization).to(device)
    optimizer = optim.Adam(model_1d.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(params=model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

    ## Train the model
    train_loss_lst, eval_loss_lst = train_model(model_1d, optimizer, train_dataloader, test_dataloader, num_epochs)
    # loss_plot(train_loss_lst, eval_loss_lst, f"hyperparameter_tuning_hidden_size_{hidden_sizes}_lr_{learning_rate:.4}_epochs_{num_epochs}_new.png")
    return eval_loss_lst[-1] # last epoch loss

def plot_search(results, x_str, y_str, res_str, scale=False):
    # Assuming coarse_results is a list of dictionaries with 'lr', 'hidden_size', 'val_loss', and 'accuracy'
    # Extract relevant information for the heatmap
    lr_values = [result[x_str] for result in results]
    hidden_size_values = [result[y_str] for result in results]
    val_loss_values = [result[res_str] for result in results]

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    heatmap = plt.scatter(lr_values, hidden_size_values, c=val_loss_values, cmap='RdYlGn', marker='o', s=100)
    plt.colorbar(heatmap, label=res_str)
    if scale:
        plt.xscale('log')  # Use a logarithmic scale for learning rates if appropriate

    # Set labels and title
    plt.xlabel(x_str)
    plt.ylabel(y_str)
    plt.title('Hyperparameter Search')
    plt.grid(True)

    # Show the plot
    plt.savefig("comparison_coarse_fine_hyperparameter_tuning_new.png")

    plt.show()
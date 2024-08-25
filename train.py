import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    
    return {
        "Mean Absolute Error (MAE)": mae,
        "Mean Squared Error (MSE)": mse,
        "Root Mean Squared Error (RMSE)": rmse,
        "R-squared (R^2)": r2
    }

def loss_plot(train_loss, validation_loss, filename):
    epochs = range(1, len(train_loss) + 1) # start at 1 instead of 0
    # Plotting the training and validation losses
    plt.figure(figsize=(5, 5))
    plt.plot(epochs, train_loss, label='Training Loss', color='blue')
    plt.plot(epochs, validation_loss, label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename)
    plt.show()

def train_model(net, optimizer, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define the loss function
    criterion = nn.MSELoss()
    # Define the optimizer

    train_loss_lst = []
    val_loss_lst = []

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        # Iterate over training batches
        for audio, valence in train_loader:
            optimizer.zero_grad()  # Reset gradients
            valence = valence.float() # was double 

            audio = audio.unsqueeze(1) # [batch, channel=1, 128,145]
            audio, valence = audio.to(device), valence.to(device)

            outputs = net(audio)
            outputs = outputs.squeeze()  # Reshape the output to match target
            loss = criterion(outputs, valence)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_loss_lst.append(train_loss)

        # Validation
        net.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for audio, valence in val_loader:
                valence = valence.float() # was double

                audio = audio.unsqueeze(1) # [batch, channel=1, 128,145]
                audio, valence = audio.to(device), valence.to(device)
                outputs = net(audio)
                outputs = outputs.squeeze()  # Reshape the output to match target
                val_loss = criterion(outputs, valence)
                val_running_loss += val_loss.item()

        val_loss = val_running_loss / len(val_loader)
        val_loss_lst.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
    #display loss graph
    # loss_plot(train_loss_lst, val_loss_lst, "")
    print("Training for CNN is finished.")

    return train_loss_lst, val_loss_lst
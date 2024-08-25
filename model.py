import torch
import torch.nn as nn



class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))

    def forward(self, x):
        with torch.no_grad():
            x = x - self.mean
            x = x / self.std
        return x
    

class CNN1d(nn.Module):
    def __init__(self, pre_proocesses, hidden_sizes, activation_function, norm_layer):
        # from main import normalization 
        super(CNN1d, self).__init__()

        self.hidden_sizes = hidden_sizes
        self.activation_function = activation_function

        self.width = 128

        self.layers = nn.ModuleList()

        # # add preprocessing steps
        # for process in pre_proocesses:
        #     self.layers.append(process)
        self.layers.append(norm_layer)
        ############################################ stopped here need to get normalization from above to class
        for i in range(len(self.hidden_sizes)):
            self.layers.append(nn.Conv1d(1 if i ==0 else self.hidden_sizes[i-1], self.hidden_sizes[i], kernel_size=3))
            self.layers.append(nn.BatchNorm1d(self.hidden_sizes[i], eps=.00001, momentum=0.1, affine=True, track_running_stats=True))
            self.layers.append(nn.MaxPool1d(kernel_size=3))
            self.layers.append(self.activation_function())


        self.layers.append(nn.AdaptiveAvgPool1d(1))
        self.layers.append(nn.Flatten()) 
        self.layers.append(nn.Linear(in_features=self.hidden_sizes[-1], out_features=self.width))
        self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nn.Linear(in_features=self.width, out_features=1))

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
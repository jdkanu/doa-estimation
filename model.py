import torch.nn as nn
import torch

LSTM_FULL = "Full"
LSTM_FIRST = "First"
LSTM_LAST = "Last"

class CRNN(nn.Module):
    def __init__(self, device, dropouts, output_dimension, is_classifier, lstm_out_form):
        super(CRNN, self).__init__()
        self.lstm_out_form = lstm_out_form
        self.device = device
        self.dropouts = dropouts
        conv_layers = [
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 8)),  # (bsz, 64, 25, 64)

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 8)),  # (bsz, 64, 25, 8)

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),  # (bsz, 64, 25, 2)
        ]
        modules = []
        for layer in conv_layers:
            modules.append(layer)
            modules.append(dropouts.conv_dropout)
        self.conv = nn.Sequential(*modules)

        # from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/bidirectional_recurrent_neural_network/main.py
        self.hidden_size = 64
        self.num_layers = 2
        self.lstm = nn.LSTM(input_size=128, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(self.hidden_size*2, self.hidden_size*2)
        self.fc2 = nn.Linear(self.hidden_size*2, output_dimension)

        softmax_dim = 2 if self.lstm_out_form == LSTM_FULL else 1
        self.softmax = nn.Softmax(softmax_dim) if is_classifier else None


    def forward(self, x):
        out = self.dropouts.input_dropout(x)
        out = self.conv(out)  # (bsz, 64, 25, 2)
        reshape = out.permute(0, 2, 1, 3).contiguous().view(len(out), 25, 128)
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, reshape.size(0), self.hidden_size).to(self.device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, reshape.size(0), self.hidden_size).to(self.device)

        lstm_out, _ = self.lstm(reshape, (h0, c0))

        if self.lstm_out_form == LSTM_FULL:
            fc_out = self.fc2(self.fc1(lstm_out))
            return self.softmax(fc_out).permute(0, 2, 1) if self.softmax else fc_out
        else:
            lstm_out_ind = 0 if self.lstm_out_form == LSTM_FIRST else -1
            fc_out = self.fc2(self.fc1(lstm_out[:, lstm_out_ind, :]))
            return self.softmax(fc_out) if self.softmax else fc_out


class ConvNet(nn.Module):
    def __init__(self, device, dropouts, output_dimension, is_classifier):
        super(ConvNet, self).__init__()
        self.device = device
        self.dropouts = dropouts
        conv_layers = [
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 8)),  # (bsz, 64, 25, 64)

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 8)),  # (bsz, 64, 25, 8)

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),  # (bsz, 64, 25, 2)
        ]
        modules = []
        for layer in conv_layers:
            modules.append(layer)
            modules.append(dropouts.conv_dropout)
        self.conv = nn.Sequential(*modules)
        self.fc1 = nn.Linear(64*25*2, 64*25*2)
        self.fc2 = nn.Linear(64*25*2, output_dimension)
        #self.fc = nn.Linear(64*25*2, output_dimension)
        self.softmax = nn.Softmax(1) if is_classifier else None

    def forward(self, x):
        out = self.dropouts.input_dropout(x)
        out = self.conv(out)  # (bsz, 64, 25, 2)
        flattened = out.view(len(out), 64*25*2)
        #fc_out = self.fc(flattened)
        fc_out = self.fc2(self.fc1(flattened))
        return self.softmax(fc_out) if self.softmax else fc_out

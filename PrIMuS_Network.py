import torch.nn as nn
from collections import OrderedDict

class PrIMuS_Network(nn.Module):
    def __init__(self, rnn_hidden, leaky_relu, num_class):
        super(PrIMuS_Network, self).__init__()

        self.leaky_relu = leaky_relu
        self.cnn = self._cnn_backbone()

        # channel * height
        cnn_output_layer = 256 * 6
        
        # self.map_to_seq = nn.Linear(cnn_output_layer, map_to_seq_hidden)

        self.rnn1 = nn.LSTM(cnn_output_layer, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        # rnn hidden to output
        self.dense = nn.Linear(2 * rnn_hidden, num_class)

    def _cnn_backbone(self):
        cnn = nn.Sequential(OrderedDict([
            ('bn1', nn.BatchNorm2d(1)),
            ('ReLU1', nn.LeakyReLU(self.leaky_relu, inplace=True)),
            ('conv1', nn.Conv2d(1, 32, 3)),
            ('pooling1', nn.MaxPool2d(2, stride=2)),
            ('bn2', nn.BatchNorm2d(32)),
            ('ReLU2', nn.LeakyReLU(self.leaky_relu, inplace=True)),
            ('conv2', nn.Conv2d(32, 64, 3)),
            ('pooling2', nn.MaxPool2d(2, stride=2)),
            ('bn3', nn.BatchNorm2d(64)),
            ('ReLU3', nn.LeakyReLU(self.leaky_relu, inplace=True)),
            ('conv3', nn.Conv2d(64, 128, 3)),
            ('pooling3', nn.MaxPool2d(2, stride=2)),
            ('bn4', nn.BatchNorm2d(128)),
            ('ReLU4', nn.LeakyReLU(self.leaky_relu, inplace=True)),
            ('conv4', nn.Conv2d(128, 256, 3)),
            ('pooling4', nn.MaxPool2d(2, stride=2)),
        ]))
        return cnn

    def forward(self, input):
        conv = self.cnn(input)
        batch, channel, height, width = conv.size()

        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(2, 0, 1)  # (width, batch, feature)
        
        # seq = self.map_to_seq(conv)

        recurrent, _ = self.rnn1(conv)
        recurrent, _ = self.rnn2(recurrent)

        output = self.dense(recurrent)

        return output
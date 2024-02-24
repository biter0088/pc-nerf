import torch
import torch.nn as nn

class Embedding(nn.Module):
    """
    Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)

    :param in_channels: number of input channels (2 for both xy) 
                                                
    :param N_freq: number of scale of embeddings
    :param logscale: whether to use log scale (default: True)

    """

    def __init__(self, in_channels, N_freq, logscale=True):
        super(Embedding, self).__init__()
        self.N_freq = N_freq
        self.in_channels = in_channels

        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freq - 1, N_freq)            
        else:
            self.freq_bands = torch.linspace(1, 3 ** (N_freq - 1), N_freq)            # 

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Embeds x to ( sin(2^k x), cos(2^k x), ...)-

        :param x: (B, self.in_channels)
        :return out: (B, self.N_freq * self.in_channels * len(self.funcs))
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out.append(func(freq * x))

        out = torch.cat(out, -1)
        return out


class NOF(nn.Module):
    def __init__(self, feature_size=256, in_channels_xy=63, use_skip=True):  #       
        """
        The model of NOF.

        :param feature_size: number of hidden units in each layer  
        :param in_channels_xy: number of input channels for xy (default: 2+2*10*2=42)
                        in_channels_xyz: number of input channels for xyz (default: 3+3*10*2=63)  # 
                        in_channels_xyz: number of input channels for xyz (default: 3*10*2=63)  #   
        :param use_skip: whether to use skip architecture (default: True)
        """
        super(NOF, self).__init__()
        self.feature_size = feature_size
        self.in_channels_xy = in_channels_xy
        self.use_skip = use_skip

        # define nerf model
        # layer 1: first 4-layers MLP
        self.layer1 = []
        for i in range(4):
            if i == 0:
                self.layer1.append(
                    nn.Linear(in_features=self.in_channels_xy, out_features=self.feature_size))
            else:
                self.layer1.append(
                    nn.Linear(in_features=self.feature_size, out_features=self.feature_size))
            self.layer1.append(nn.BatchNorm1d(num_features=self.feature_size))
            # self.layer1.append(nn.ReLU(True))
            self.layer1.append(nn.LeakyReLU(True))#

        self.layer1 = nn.Sequential(*self.layer1)

        # layer 2: second 4-layers MLP
        self.layer2 = []
        for i in range(4):
            if i == 0:
                if self.use_skip:
                    self.layer2.append(
                        nn.Linear(in_features=self.in_channels_xy + self.feature_size,
                                  out_features=self.feature_size))
                else:
                    self.layer2.append(
                        nn.Linear(in_features=self.feature_size, out_features=self.feature_size))
            else:
                self.layer2.append(
                    nn.Linear(in_features=self.feature_size, out_features=self.feature_size))
            self.layer2.append(nn.BatchNorm1d(num_features=self.feature_size))
            # self.layer1.append(nn.ReLU(True))
            self.layer1.append(nn.LeakyReLU(True))#

        self.layer2 = nn.Sequential(*self.layer2)

        # occupancy probability
        self.occ_out = nn.Sequential(
            nn.Linear(in_features=self.feature_size, out_features=1),
            nn.Sigmoid()#
        )


    def forward(self, x):
        """
        Encodes input position (xy) to occupancy probability (p_occ)  
        :param x: the embedded vector of a 2D position
                  (shape: (B, self.in_channels_xy))

        :return p_occ: the occupancy probability of the input position (shape: (B, 1))
        """
        # print("x.shape\n",x.shape)
        input_xy = x

        feature1 = self.layer1(input_xy)

        if self.use_skip:
            feature1 = torch.cat([input_xy, feature1], dim=1)

        feature2 = self.layer2(feature1)

        p_occ = self.occ_out(feature2)

        return p_occ

class NOF_coarse(nn.Module):
    def __init__(self, feature_size=256, in_channels_xy=63, use_skip=True):  
        """
        The model of NOF_coarse.

        :param feature_size: number of hidden units in each layer  
        :param in_channels_xy: number of input channels for xy (default: 2+2*10*2=42)
                        in_channels_xyz: number of input channels for xyz (default: 3+3*10*2=63)  
        :param use_skip: whether to use skip architecture (default: True)
        """
        super(NOF_coarse, self).__init__()
        self.feature_size = feature_size
        self.in_channels_xy = in_channels_xy
        self.use_skip = use_skip

        # define nerf model
        # layer 1: first 4-layers MLP
        self.layer1 = []
        for i in range(4):
            if i == 0:
                self.layer1.append(
                    nn.Linear(in_features=self.in_channels_xy, out_features=self.feature_size))
            else:
                self.layer1.append(
                    nn.Linear(in_features=self.feature_size, out_features=self.feature_size))
            self.layer1.append(nn.BatchNorm1d(num_features=self.feature_size))
            # self.layer1.append(nn.ReLU(True))
            self.layer1.append(nn.LeakyReLU(True))#

        self.layer1 = nn.Sequential(*self.layer1)

        # layer 2: second 4-layers MLP
        self.layer2 = []
        for i in range(4):
            if i == 0:
                if self.use_skip:
                    self.layer2.append(
                        nn.Linear(in_features=self.in_channels_xy + self.feature_size,
                                  out_features=self.feature_size))
                else:
                    self.layer2.append(
                        nn.Linear(in_features=self.feature_size, out_features=self.feature_size))
            else:
                self.layer2.append(
                    nn.Linear(in_features=self.feature_size, out_features=self.feature_size))
            self.layer2.append(nn.BatchNorm1d(num_features=self.feature_size))
            # self.layer1.append(nn.ReLU(True))
            self.layer1.append(nn.LeakyReLU(True))#

        self.layer2 = nn.Sequential(*self.layer2)

        # occupancy probability
        self.occ_out = nn.Sequential(
            nn.Linear(in_features=self.feature_size, out_features=1),
            nn.Sigmoid()#
        )


    def forward(self, x):
        """
        Encodes input position (xy) to occupancy probability (p_occ)  
        :param x: the embedded vector of a 2D position
                  (shape: (B, self.in_channels_xy))

        :return p_occ: the occupancy probability of the input position (shape: (B, 1))
        """
        # print("x.shape\n",x.shape)
        input_xy = x

        feature1 = self.layer1(input_xy)

        if self.use_skip:
            feature1 = torch.cat([input_xy, feature1], dim=1)

        feature2 = self.layer2(feature1)

        p_occ = self.occ_out(feature2)

        return p_occ

class NOF_fine(nn.Module):
    def __init__(self, feature_size=256, in_channels_xy=63, use_skip=True):  #       
        """
        The model of NOF_fine.

        :param feature_size: number of hidden units in each layer  
        :param in_channels_xy: number of input channels for xy (default: 2+2*10*2=42)
                        in_channels_xyz: number of input channels for xyz (default: 3+3*10*2=63)  #
        :param use_skip: whether to use skip architecture (default: True)
        """
        super(NOF_fine, self).__init__()
        self.feature_size = feature_size
        self.in_channels_xy = in_channels_xy
        self.use_skip = use_skip

        # define nerf model
        # layer 1: first 4-layers MLP
        self.layer1 = []
        for i in range(4):
            if i == 0:
                self.layer1.append(
                    nn.Linear(in_features=self.in_channels_xy, out_features=self.feature_size))
            else:
                self.layer1.append(
                    nn.Linear(in_features=self.feature_size, out_features=self.feature_size))
            self.layer1.append(nn.BatchNorm1d(num_features=self.feature_size))
            # self.layer1.append(nn.ReLU(True))
            self.layer1.append(nn.LeakyReLU(True))#

        self.layer1 = nn.Sequential(*self.layer1)

        # layer 2: second 4-layers MLP
        self.layer2 = []
        for i in range(4):
            if i == 0:
                if self.use_skip:
                    self.layer2.append(
                        nn.Linear(in_features=self.in_channels_xy + self.feature_size,
                                  out_features=self.feature_size))
                else:
                    self.layer2.append(
                        nn.Linear(in_features=self.feature_size, out_features=self.feature_size))
            else:
                self.layer2.append(
                    nn.Linear(in_features=self.feature_size, out_features=self.feature_size))
            self.layer2.append(nn.BatchNorm1d(num_features=self.feature_size))
            # self.layer1.append(nn.ReLU(True))
            self.layer1.append(nn.LeakyReLU(True))#

        self.layer2 = nn.Sequential(*self.layer2)

        # occupancy probability
        self.occ_out = nn.Sequential(
            nn.Linear(in_features=self.feature_size, out_features=1),
            nn.Sigmoid()#
        )

    def forward(self, x):
        """
        Encodes input position (xy) to occupancy probability (p_occ)  
        :param x: the embedded vector of a 2D position
                  (shape: (B, self.in_channels_xy))

        :return p_occ: the occupancy probability of the input position (shape: (B, 1))
        """
        # print("x.shape\n",x.shape)
        input_xy = x

        feature1 = self.layer1(input_xy)

        if self.use_skip:
            feature1 = torch.cat([input_xy, feature1], dim=1)

        feature2 = self.layer2(feature1)

        p_occ = self.occ_out(feature2)

        return p_occ

class NOF_plusfine(nn.Module):
    def __init__(self, feature_size=256, in_channels_xy=63, use_skip=True):  
        """
        The model of NOF_fine.

        :param feature_size: number of hidden units in each layer  
        :param in_channels_xy: number of input channels for xy (default: 2+2*10*2=42)
                        in_channels_xyz: number of input channels for xyz (default: 3+3*10*2=63)  z
        :param use_skip: whether to use skip architecture (default: True)
        """
        super(NOF_plusfine, self).__init__()
        self.feature_size = feature_size
        self.in_channels_xy = in_channels_xy
        self.use_skip = use_skip

        # define nerf model
        # layer 1: first 4-layers MLP
        self.layer1 = []
        for i in range(4):
            if i == 0:
                self.layer1.append(
                    nn.Linear(in_features=self.in_channels_xy, out_features=self.feature_size))
            else:
                self.layer1.append(
                    nn.Linear(in_features=self.feature_size, out_features=self.feature_size))
            self.layer1.append(nn.BatchNorm1d(num_features=self.feature_size))
            # self.layer1.append(nn.ReLU(True))
            self.layer1.append(nn.LeakyReLU(True))#

        self.layer1 = nn.Sequential(*self.layer1)

        # layer 2: second 4-layers MLP
        self.layer2 = []
        for i in range(4):
            if i == 0:
                if self.use_skip:
                    self.layer2.append(
                        nn.Linear(in_features=self.in_channels_xy + self.feature_size,
                                  out_features=self.feature_size))
                else:
                    self.layer2.append(
                        nn.Linear(in_features=self.feature_size, out_features=self.feature_size))
            else:
                self.layer2.append(
                    nn.Linear(in_features=self.feature_size, out_features=self.feature_size))
            self.layer2.append(nn.BatchNorm1d(num_features=self.feature_size))
            # self.layer1.append(nn.ReLU(True))
            self.layer1.append(nn.LeakyReLU(True))#

        self.layer2 = nn.Sequential(*self.layer2)

        # occupancy probability
        self.occ_out = nn.Sequential(
            nn.Linear(in_features=self.feature_size, out_features=1),
            nn.Sigmoid()#
        )

    def forward(self, x):
        """
        Encodes input position (xy) to occupancy probability (p_occ)  
        :param x: the embedded vector of a 2D position
                  (shape: (B, self.in_channels_xy))

        :return p_occ: the occupancy probability of the input position (shape: (B, 1))
        """
        input_xy = x

        feature1 = self.layer1(input_xy)

        if self.use_skip:
            feature1 = torch.cat([input_xy, feature1], dim=1)

        feature2 = self.layer2(feature1)

        p_occ = self.occ_out(feature2)

        return p_occ    
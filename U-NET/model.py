import torch
import torch.nn as nn
from torchvision.transforms.functional import center_crop


# performs a convolution the applies batch normalization and RELU
class CNNBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0):
        super(CNNBlock, self).__init__()

        self.seq_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.seq_block(x)
        return x


# creates and performs a block of convolution
class CNNBlocks(nn.Module):
    """
    Parameters:
    n_conv (int): creates a block of n_conv convolutions
    in_channels (int): number of in_channels of the first block's convolution
    out_channels (int): number of out_channels of the first block's convolution
    expand (bool) : if True after the first convolution of a blocl the number of channels doubles
    """
    def __init__(self,
                 n_conv,
                 in_channels,
                 out_channels,
                 padding):
        super(CNNBlocks, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(n_conv):

            self.layers.append(CNNBlock(in_channels, out_channels, padding=padding))
            # after each convolution we set (next) in_channel to (previous) out_channels
            in_channels = out_channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# performs the encoding of the UNET with 5 blocks of CNNBlocks and stores the routes to be
# concatenated channel-wise to the decoder
# https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
class Encoder(nn.Module):
    """
    Parameters:
    in_channels (int): number of in_channels of the first CNNBlocks
    out_channels (int): number of out_channels of the first block's convolution
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 padding):
        super(Encoder, self).__init__()
        self.enc_layers = nn.ModuleList()
        # we add the first block "manually" since we are not using expand=True
        self.enc_layers += [CNNBlocks(n_conv=2, in_channels=in_channels,
                                      out_channels=out_channels, padding=padding),
                            nn.MaxPool2d(2, 2)]
        # after each block we set (next) in_channel to (previous) out_channel
        in_channels = out_channels
        # we add three blocks with expand=True
        for _ in range(3):
            # increasing the depth of the first convolution's block
            out_channels *= 2
            self.enc_layers += [
                    CNNBlocks(n_conv=2, in_channels=in_channels, out_channels=out_channels, padding=padding),
                    nn.MaxPool2d(2, 2)
                ]

            in_channels = out_channels
        # doubling the dept of the last CNN block
        out_channels *= 2
        self.enc_layers.append(CNNBlocks(n_conv=2, in_channels=in_channels,
                                         out_channels=out_channels, padding=padding))

    def forward(self, x):
        route_connection = []
        for layer in self.enc_layers:
            if isinstance(layer, CNNBlocks):
                x = layer(x)
                route_connection.append(x)
            else:
                x = layer(x)
        return x, route_connection


class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 exit_channels,
                 padding):
        super(Decoder, self).__init__()
        self.exit_channels = exit_channels
        self.layers = nn.ModuleList()
        # the out channel will be fed to the decoder so that out_channels = in_channels // 2
        # i.e (1024, 512)
        self.layers += [
            # # i.e (1024, 512)
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        ]

        for i in range(3):
            # first block of CNNBlocks will still be out_channels = in_channels // 2
            # inside the CNNBlocks after the first convolution out_channels will be set equal to in_channels
            # however in_channels will be reduced in the subclass but not here, hence in to match out_channels,
            # the in_channels of the transpose conv have to be set to in_channel//2
            self.layers += [
                # i.e (1024, 512) --> (512, 512)
                CNNBlocks(n_conv=2, in_channels=in_channels,
                          out_channels=out_channels, padding=padding),
                # i.e (512, 256)
                nn.ConvTranspose2d(in_channels//2, out_channels//2, kernel_size=2, stride=2)
            ]
            # the out_channel of the transposed_conv are half of the subsequent in_channels
            # since the outputs of the encoder are being concatenated making dimensions match
            # channel-wise to the in_channels.
            # i.e (512, 256)
            in_channels //= 2
            out_channels //= 2

        self.layers += [
            CNNBlocks(n_conv=2, in_channels=in_channels, out_channels=out_channels, padding=padding),
            # out layer
            CNNBlock(in_channels//2, self.exit_channels, padding=1)
        ]

    def forward(self, x, routes_con):
        # pop the last element of the list since
        # it's not used for concatenation
        routes_con.pop(-1)
        for layer in self.layers:
            if isinstance(layer, CNNBlocks):
                # center_cropping the route tensor to make width and height match
                routes_con[-1] = center_crop(routes_con[-1], x.shape[2])
                # concatenating tensors channel-wise
                x = torch.cat([x, routes_con.pop(-1)], dim=1)
                x = layer(x)
            else:
                x = layer(x)
        return x


# bad practices:
# 1) out_channels should define the out_channels of the output layer,
# while in my case they are just referring to the out_channel of the first convolution
# 2) modifying the forward() of the decoder is a bad practice. I could have got the same result
# by appending route_connections in the forward method of the UNET.
# 3) Lastly as shown in the UNET by Aladdin, here encoder and decoder are not necessary:
# I could have simply included them in the UNET class itself

class UNET(nn.Module):
    def __init__(self,
                 in_channels,
                 expansion,
                 exit_channels,
                 padding=0):
        super(UNET, self).__init__()
        self.encoder = Encoder(in_channels, expansion, padding=padding)
        self.decoder = Decoder(expansion*16, expansion*8, exit_channels, padding=padding)

    def forward(self, x):
        enc_out, routes = self.encoder(x)
        out = self.decoder(enc_out, routes)
        return out


if __name__ == "__main__":

    unet = UNET(1, 64, 1, padding=0)
    test = torch.rand((3, 1, 572, 572))
    print("out unet shape is", unet(test).shape)







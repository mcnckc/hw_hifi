from torch import nn
from hw_hifi.base import BaseModel
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

def init_weights(m, mean=0.0, std=0.01):
    if "Conv" in m.__class__.__name__:
        m.weight.data.normal_(mean, std)

class ResBlock(nn.Module):
    def __init__(self, n_channels, ks=3, dilations=[1, 3, 5]):
        super().__init__()
        self.layers = nn.ModuleList()
        for d in dilations:
            block = nn.Sequential(
                nn.LeakyReLU(0.1),
                weight_norm(nn.Conv1d(n_channels, n_channels, kernel_size=ks, dilation=d, padding='same')),
                nn.LeakyReLU(0.1),
                weight_norm(nn.Conv1d(n_channels, n_channels, kernel_size=ks, padding='same'))
            )
            block.apply(init_weights)
            self.layers.append(block)

    def forward(self, x):
        return x + sum([layer(x) for layer in self.layers])
    

class MRF(nn.Module):
    def __init__(self, n_channels, kernels, dilations):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.n_blocks = len(kernels)
        for ks, d in zip(kernels, dilations):
            self.blocks.append(ResBlock(n_channels, ks, d))

    def forward(self, x):
        return sum([block(x) for block in self.blocks]) / self.n_blocks
    
class Generator(BaseModel):
    def __init__(self, init_channels, kernel_sizes, strides, res_kernels, res_dilations, **batch):
        super().__init__(**batch)
        layers = nn.ModuleList()
        layers.append(weight_norm(nn.Conv1d(80, init_channels, 7, 1, padding=3)))
        channels = init_channels
        for ks, stride in zip(kernel_sizes, strides):
            layers.extend([
                nn.LeakyReLU(0.1),
                weight_norm(nn.ConvTranspose1d(channels, channels // 2,
                                                              ks, stride, padding=(ks - stride) // 2)),
                MRF(channels // 2, res_kernels, res_dilations)
            ])
            channels //= 2 
        layers.extend([
                nn.LeakyReLU(0.1),
                weight_norm(nn.Conv1d(channels, 1, kernel_size=7, padding="same")),
                nn.Tanh(),
        ])
        self.layers = nn.ModuleList(layers)
        self.net = nn.Sequential(*layers)
        self.net.apply(init_weights)

    def forward(self, spectrogram, **batch):
        """
        print("FORWARD:")
        x = spectrogram
        for l in self.layers:
            x = l(x)
            print(l)
            if x is None:
                print("BROKEN layer", l)
        """
        return self.net(spectrogram)

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here

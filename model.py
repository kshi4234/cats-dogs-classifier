import torch
import torch.nn as nn


class Block(nn.Module):
  expansion = 4 # Set to some integer number
  def __init__(self, in_channels, out_channels, K=3, S=1, P=1, downsample=None):
    """
    Multiplies the number of output channels in the 3rd convolution by this number
    If set to 1, keeps the number of out_channels consistent within the hidden Block
    """
    super(Block, self).__init__()
    self.downsample = downsample
    # print(in_channels, out_channels, K, S, P, downsample)
    self.standard_block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=K, stride=S, padding=P, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels * 4),
    )
    self.final_relu = nn.ReLU()

  def forward(self, x):
    out = self.standard_block(x)
    residual = x
    if self.downsample is not None:
      residual = self.downsample(x)
    out += residual
    out = self.final_relu(out)
    return out

class Classifier(nn.Module):
  def __init__(self, *args, **kwargs):
    """
    H, W: The height and width of the image
    num_layers: Number of ResNet layers - NOT INTENDED TO BE USED!!!
    num_blocks: List of number of blocks in each layer
    num_channels: List of dimensions of the output channels at each layer
    num_classes: Number of classes in the data
    grayscale: Boolean of whether of not the image is in grayscale or not
    """
    super(Classifier, self).__init__()
    self.inchannels = 64  # Number of channels at the beginning of each official layer
    if 'H' in kwargs and 'W' in kwargs:
      self.H, self.W = kwargs['H'], kwargs['W']
    else:
      self.H, self.W = 224, 224
    if 'num_layers' in kwargs and kwargs['num_layers']:
      self.num_layers = kwargs['num_layers']
    else:
      self.num_layers = 4
    if 'num_channels' in kwargs:
      self.out_channels = kwargs['num_channels']
      # Ensure length matches the number of layers
      if len(self.out_channels) != self.num_layers:
        self.num_layers = len(self.out_channels)
        print('num_layers does not match length of out_channels: setting num_layers to {}'.format(self.num_layers))
    else:
      self.out_channels = [64, 128, 256, 512]
    if 'num_blocks' in kwargs:
      self.num_blocks = kwargs['num_blocks']
    else:
      self.num_blocks = [3, 4, 6, 3]  # ResNet50 block count
    if 'grayscale' in kwargs:
      self.grayscale = kwargs['grayscale']
    else:
      self.grayscale = False

    self.num_classes = kwargs['num_classes'] if 'num_classes' in kwargs else 2
    in_dims = 1 if self.grayscale else 3

    # ------------------------ Define the actual architecture ------------------------ #
    # First have an initial layer the extracts features from the image
    self.init_conv = nn.Sequential(
        nn.Conv2d(in_dims, self.inchannels, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(self.inchannels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    # Then instantiate block layers, initial keeps same size
    layers = []
    x = self.make_layer(channels=self.out_channels[0],
                  K=3, S=1, P=1,
                  num_blocks=self.num_blocks[0], block=Block)
    layers.append(x)
    for i in range(1, self.num_layers):
      x = self.make_layer(channels=self.out_channels[i],
                    K=3, S=2, P=1,
                    num_blocks=self.num_blocks[i], block=Block)
      layers.append(x)
    self.block_layers = nn.Sequential(*layers)
    # Global Average pooling at the output
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    # Linear classification layer
    self.fc = nn.Linear(self.inchannels, 1000)

    # Kaiming initialization
    for l in self.modules():
      if isinstance(l, nn.Conv2d):
        nn.init.kaiming_normal_(l.weight, mode="fan_out", nonlinearity="relu")
      elif isinstance(l, nn.BatchNorm2d):
        # Manual initialization same as default for BatchNorm2d, just include to make explicit
        l.weight.data.fill_(1)
        l.bias.data.zero_()

  def make_layer(self, channels, K, S, P, num_blocks, block=Block):
    """
    Makes a layer, which is a composition of instances of some class implementing nn.Module
    Depending on num_blocks, each layer will have some number of blocks
    """
    downsample = None
    if (S != 1) or (self.inchannels != channels * block.expansion):
      downsample = nn.Sequential(
          nn.Conv2d(in_channels=self.inchannels, out_channels=(channels * block.expansion),
                    kernel_size=1, stride=S, bias=False),
          nn.BatchNorm2d(channels * block.expansion)
      )
    layers = []
    layers.append(block(self.inchannels, channels, K=K, S=S, P=P, downsample=downsample))
    self.inchannels = channels * block.expansion
    for i in range(1, num_blocks):
      layers.append(block(self.inchannels, channels, K=3, S=1, P=1))
    return nn.Sequential(*layers)

  def forward(self, x):
    out = self.init_conv(x)
    out = self.block_layers(out)
    out = self.avgpool(out)
    out = out.view(out.shape[0], -1)
    logits = self.fc(out)
    probs = nn.functional.softmax(logits, dim=1)
    return logits, probs
import math

import torch
import torch.nn as nn
import torchvision

def conv3x3(in_channels,out_channels,stride=1):
	return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.LeakyReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self,in_channels,out_channels,stride=1,downsample=None):
		super(BasicBlock,self).__init__()
		self.conv1 = conv3x3(in_channels,out_channels,stride) #大小不變
		self.bn1 = nn.BatchNorm2d(out_channels) # 批標準化層
		self.relu = nn.LeakyReLU(inplace=True) # 激活函數

		self.conv2 = conv3x3(out_channels,out_channels) #大小不變
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.downsample = downsample # 這個是shortcut
		self.stride = stride

	def forward(self,x):
		residual = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=219):
        super(ResNet, self).__init__()
        self.in_channels = 64  #每一個殘差塊塊輸入深度

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) #224-5+4/2+1=112
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 縮小1/2
        #112-3+2/2+1=56
        self.layer1 = self._make_layer(block, 64, layers[0])  # 特徵圖大小不變
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 56-1/2+1=28
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 28-1/2+1=14
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)  # 14-1/2+1=7
        #self.layer5 = self._make_layer(block, 512, layers[4], stride=2)  # 7-1/2+1=4


        #self.avgpool = nn.AvgPool2d((7, 7))
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))  # 平均池化
        self.fc = nn.Linear(512 * 2 * 2, 1024),
        self.classifier = nn.Sequential(
            #nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

        # 網絡的參數初始化
        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None  # shortcut
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(#64,64,1,2,
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion  # 改變下面的殘差塊的輸入深度

        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.conv1(x) #224-7+6/2+1=112
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)  #112-3+2/2+1=56
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)  # 將原有的多維輸出拉回一維
        #out = self.fc(out)
        out = self.classifier(out)

        return out

def ResNet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def BResNet18(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model
'''
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=219):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False) #224-7+6/1+1=224
        self.bn1 = nn.BatchNorm2d(64)
        self.gelu = nn.GELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) #224-3+2/2+1=113
        self.layer1 = self._make_layer(block, 64, layers[0]) #113-1/1+1=113
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Sequential(
            nn.Linear(512 * block.expansion, 512 * 10),
            nn.BatchNorm1d(512 * 10),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512 * 10, 512 * 10),
            nn.BatchNorm1d(512 * 10),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512 * 10, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n)) # 卷積參數變量初始化
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1) # BN參數初始化
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion: #1
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.gelu = nn.GELU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.gelu(out)

        return out
'''
if __name__=='__main__':
    #model = torchvision.models.resnet50()
    model = ResNet18()
    #print(model)

    input = torch.randn(64, 3, 224, 224)
    out = model(input)
    print(out.shape)

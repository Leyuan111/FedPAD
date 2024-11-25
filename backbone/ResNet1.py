import torch
import torch.nn as nn
from typing import List
from torch.hub import load_state_dict_from_url

model_urls = {
    'resnet10': None,
    'resnet12': None,
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet20': None,
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
}

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 卷积层，带有填充"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 卷积层"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=bias)

class BasicBlock(nn.Module):
    """ResNet 的基本残差块（适用于 ResNet-18 和 ResNet-34）"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample  # 下采样层

    def forward(self, x):
        identity = x  # 残差连接
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 如果需要下采样，则对输入进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 残差连接
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    """Bottleneck 残差块（适用于 ResNet-50 及以上）"""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_planes, planes)  # 压缩通道数
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)  # 空间维度的卷积
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)  # 恢复通道数
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # 下采样层

    def forward(self, x):
        identity = x  # 残差连接

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 如果需要下采样，则对输入进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 残差连接
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """ResNet 网络结构"""

    def __init__(self, block, layers: List[int], num_classes, nf=64, name='ResNet'):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.name = name  # 添加 name 属性

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, nf, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(nf)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差层
        self.layer1 = self._make_layer(block, nf, layers[0])
        self.layer2 = self._make_layer(block, nf * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, layers[3], stride=2)

        # 平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(nf * 8 * block.expansion, num_classes)

        # 权重初始化
        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        """构建 ResNet 的一个层（包含多个残差块）"""
        downsample = None
        # 判断是否需要下采样
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # 第一个残差块
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        # 其余的残差块
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming 正态初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # 批归一化初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 全连接层初始化
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def features(self,x):
            # 前向传播
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # 第1层
        x = self.layer2(x)  # 第2层
        x = self.layer3(x)  # 第3层
        x = self.layer4(x)  # 第4层

        x = self.avgpool(x)  # 自适应平均池化
        x = torch.flatten(x, 1)
        return x

    def classifier(self,x):
        out = self.fc(x)
        return out


    def forward(self, x):
        # 前向传播
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # 第1层
        x = self.layer2(x)  # 第2层
        x = self.layer3(x)  # 第3层
        x = self.layer4(x)  # 第4层

        x = self.avgpool(x)  # 自适应平均池化
        x = torch.flatten(x, 1)
        x = self.fc(x)  # 全连接层

        return x

def resnet10(num_classes, nf=64):
    """构建 ResNet-10 模型"""
    model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes, nf=nf, name='resnet10')
    return model

def resnet12(num_classes, nf=64):
    """构建 ResNet-12 模型"""
    model = ResNet(BasicBlock, [2, 1, 1, 1], num_classes=num_classes, nf=nf, name='resnet12')
    return model

def resnet18(num_classes, nf=64):
    """构建 ResNet-18 模型，并加载预训练权重"""
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, nf=nf, name='resnet18')
    if model_urls['resnet18'] is not None:
        state_dict = load_state_dict_from_url(model_urls['resnet18'],
                                              progress=True)
        # 加载预训练权重，忽略最后的全连接层（因为类别数可能不同）
        pretrained_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def resnet20(num_classes, nf=64):
    """构建 ResNet-20 模型"""
    model = ResNet(BasicBlock, [1, 3, 3, 3], num_classes=num_classes, nf=nf, name='resnet20')
    return model

def resnet34(num_classes, nf=64):
    """构建 ResNet-34 模型，并加载预训练权重"""
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, nf=nf, name='resnet34')
    if model_urls['resnet34'] is not None:
        state_dict = load_state_dict_from_url(model_urls['resnet34'],
                                              progress=True)
        # 加载预训练权重，忽略最后的全连接层
        pretrained_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def resnet50(num_classes, nf=64):
    """构建 ResNet-50 模型，并加载预训练权重"""
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, nf=nf, name='resnet50')
    if model_urls['resnet50'] is not None:
        state_dict = load_state_dict_from_url(model_urls['resnet50'],
                                              progress=True)
        # 加载预训练权重，忽略最后的全连接层
        pretrained_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

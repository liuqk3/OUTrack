import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


class ResNetBackbone(nn.Module):
    def __init__(self, layer=34, fix_block=2, output_dim=512, im_info=None, pretrained=True):
        super(ResNetBackbone, self).__init__()
        self.base_net = 'resnet'
        self.layer = layer
        self.fix_block = fix_block
        self.output_dim = output_dim
        self.pretrained = pretrained

        self.im_info = im_info

        self._init_modules()

    def _init_modules(self):
        # ----------- build base cnn for feature extraction ---------------
        resnet = eval(self.base_net + '{}()'.format(self.layer))
        if self.pretrained:
            resnet_weight = model_zoo.load_url(model_urls['resnet{}'.format(self.layer)])
            resnet.load_state_dict(resnet_weight)
            print('load pretrained resnet{}'.format(self.layer) + ' model from Pytorch model zoo...')

        # Build resnet. We also add a conv layer to change the channels of the final feature map.

        num_channel = 2048
        if self.layer == 18 or self.layer == 34:
            num_channel = 512

        basenet = [   resnet.conv1,
                        resnet.bn1,
                        resnet.relu,
                        resnet.maxpool,
                        resnet.layer1,
                        resnet.layer2,
                        resnet.layer3,
                        resnet.layer4,
                        nn.Conv2d(in_channels=num_channel, out_channels=256,#self.output_dim,
                                  kernel_size=1, stride=1, padding=0),
                        resnet.relu
                        ]
        self.basenet = nn.Sequential(*basenet)
        self.avgpool = nn.AdaptiveAvgPool2d((4, 2))
        self.fc = nn.Linear(in_features=4*2*256, out_features=self.output_dim)
        # Fix some layers
        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        # self.basenet.apply(set_bn_fix)

        assert (0 <= self.fix_block <= 4) # set this value to 0, so we can train all blocks
        if self.fix_block >= 4: # fix all blocks
            for p in self.basenet[7].parameters(): p.requires_grad = False
            self.basenet[7].apply(set_bn_fix)
        if self.fix_block >= 3: # fix first 3 blocks
            for p in self.basenet[6].parameters(): p.requires_grad = False
            self.basenet[6].apply(set_bn_fix)
        if self.fix_block >= 2: # fix first 2 blocks
            for p in self.basenet[5].parameters(): p.requires_grad = False
            self.basenet[5].apply(set_bn_fix)
        if self.fix_block >= 1: # fix first 1 block
            for p in self.basenet[4].parameters(): p.requires_grad = False
            self.basenet[4].apply(set_bn_fix)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode
            self.basenet.eval()

            index_all = [7, 6, 5, 4]
            index_train = index_all[0:4-self.fix_block]

            # index = {
            #     0: [4,5,6,7],
            #     1: [5, 6, 7],
            #     2: [6, 7],
            #     3: [7],
            #     4: [],
            # }
            # index = index[self.fix_block]

            for train_block in index_train:
                self.basenet[train_block].train()
            self.basenet[8].train() # the last conv layer

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            for fix_block in index_all:
                if fix_block not in index_train:
                    self.basenet[fix_block].apply(set_bn_eval)
            # self.basenet.apply(set_bn_eval)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = self.basenet(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) # [bs, 2048]
        x = self.fc(x)
        #x = x.view(x.size(0), -1) # [bs, output_dim]

        return x
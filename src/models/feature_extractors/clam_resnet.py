import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

class CLAMResNet50(nn.Module):
    """ResNet-50 truncated at layer3 (1024-dim features), with optional CLAM-style freezing."""
    expansion = 4

    def __init__(self, pretrained: bool = True, fine_tune_stage: bool = False):
        super().__init__()
        self.inplanes = 64

        # --- Stem ---
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # --- Layers ---
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)

        # CLAM baseline stops here
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 1024

        # --- Init weights ---
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # --- Load pretrained weights if requested ---
        if pretrained:
            url = "https://download.pytorch.org/models/resnet50-19c8e357.pth"
            state_dict = model_zoo.load_url(url)
            self.load_state_dict(state_dict, strict=False)

        # --- Optional freezing ---
        if fine_tune_stage:
            self._freeze_clam_style()

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * self.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )
        layers = [self._make_block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * self.expansion
        for _ in range(1, blocks):
            layers.append(self._make_block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_block(self, inplanes, planes, stride=1, downsample=None):
        return Bottleneck_Baseline(inplanes, planes, stride, downsample)

    def _freeze_clam_style(self):
        """Freeze all params except layer2.3+ and layer3 (except BN layers)."""
        for name, param in self.named_parameters():
            if ("layer3" in name) or ("layer2.3" in name):
                if "bn" not in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class Bottleneck_Baseline(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

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
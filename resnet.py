import torch
import torch.nn as nn
import torch.nn.functional as F

# Residual Block 구조 정의
class BasicBlock(nn.Module):
        # mul은 추후 ResNet18, 34, 50, 101, 152 의 구조 생성에 사용될것이다. basic은 1이고 bottleneck은 4
        # 또한 basic block은 위 ResNet중 18, 34에만 쓰이고, 한 conv층에 두개의 conv층이 들어간다는점
    mul = 1 
    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        
        # stride를 통해 너비와 높이 조정
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        
        # stride = 1, padding = 1, 커널은 3*3 이므로 너비와 높이는 항시 유지된다.
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        
        # x를 그대로 더해주기 위함 (논문에서 F + x 에 해당하는 부분이 shortcut : element-wise addition)
        self.shortcut = nn.Sequential()
        
        # 만약 size가 안맞아 합연산이 불가하다면, 연산 가능하도록 모양을 맞춰줌
        if stride != 1: # x와 
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x) # 필요에 따라 layer를 Skip할 수 있게끔
        out = F.relu(out)
        return out
    
class BottleNeck(nn.Module):    # ResNet18, 34 등 얕은 구조의 경우 위 Basic Block으로 충분하지만, 
                                # 층이 50부터는 구조가 BottleNeckArchitecture 로 바뀌게 됨
                                # 또한 한 층에 두개의 conv에서 세개의 conv로 바뀌게 됨
    mul = 4  # 논문의 구조를 참고하면 64 -> 256 // 128 -> 512 // 256 -> 1024 // 512 -> 2048 이라서
    def __init__(self, in_planes, out_planes, stride=1):
        super(BottleNeck, self).__init__()
        
        #첫 Convolution은 너비와 높이 downsampling. 여기서 in_planes는 뒤에서 64로 받아 넣어줌
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)       # filter : 2층에서 [ 1x1, 64 ] 
        
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)       # filter : [ 3x3, 64 ]
        
        self.conv3 = nn.Conv2d(out_planes, out_planes*self.mul, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes*self.mul)    # filter : [ 1x1, 64*4=256 ]
        
        self.shortcut = nn.Sequential()
        
        # 만약 size가 안맞아 합연산이 불가하거나, 우리가 원하는 output 차원이 안나온다면 억지로 모양을 맞춰줌
        if stride != 1 or in_planes != out_planes*self.mul:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes*self.mul, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes*self.mul)
            )
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
        
        
        
    
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        #RGB 3개채널에서 64개의 Kernel(filter) 사용 (ImageNet에 사용된 ResNet이라서)
        self.in_planes = 64         # input = (224*224*3)
        
        ## Resnet 논문 ImageNet 구조 그대로 구현 
        # 논문에서 conv1
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding = 3)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # output -> (112*112*32)
        
        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1) # 논문에서 conv2 -> (56*56*128)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)# 논문에서 conv3 -> (28*28*512)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)# 논문에서 conv4 -> (14*14*1024)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)# 논문에서 conv5 -> (7*7*2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512 * block.mul, num_classes)
        
    def make_layer(self, block, out_planes, num_blocks, stride):
        # layer 앞부분에서만 크기를 절반으로 줄이므로, 아래와 같은 구조
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for i in range(num_blocks):
            layers.append(block(self.in_planes, out_planes, strides[i]))
            self.in_planes = block.mul * out_planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.maxpool1(out)
        out = self.layer1(out) # 논문에서 conv2
        out = self.layer2(out) # 논문에서 conv3
        out = self.layer3(out) # 논문에서 conv4
        out = self.layer4(out) # 논문에서 conv5
        out = self.avgpool(out)
        out = torch.flatten(out,1)
        out = self.linear(out)
        return out
    
    
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet50():
    return ResNet(BottleNeck, [3, 4, 6, 3])

def ResNet101():
    return ResNet(BottleNeck, [3, 4, 23, 3])

def ResNet152():
    return ResNet(BottleNeck, [3, 8, 36, 3])
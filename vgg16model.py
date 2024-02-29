import torch
from torch import nn
import torchvision

'''
import paramiko

ip = "140.123.97.173"
username = "EPARC2"
password = "EnSCLKdB"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(ip, port=2223, username=username, password=password, timeout=600)
client.exec_command('echo P@ssw0rd | sudo -S reboot')
print("Command have been completed")
'''
class myVGG16(nn.Module):
    def __init__(self, num_classes=219):
        super(myVGG16, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # (224+4-11)/4+1 = 55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 55-3+2*0/2+1 = 27
            nn.Conv2d(64, 192, kernel_size=5, padding=2),  # 55+2-5/1+1 = 52
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # 52+2-3/1+1 = 51
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 51+2-3/1+1 = 50
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 50+2-3/1+1 = 49
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 49-3/2+1 = 24
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.myLinear = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.myLinear(x)
        return x

if __name__ == '__main__':
    mvgg16 = myVGG16()
    input1 = torch.ones((1, 3, 224, 224))
    output1 = mvgg16(input1)
    print(output1.shape)



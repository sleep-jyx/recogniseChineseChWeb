import torch
import torch.nn as nn


class NetSmall(nn.Module):
    def __init__(self):
        super(NetSmall, self).__init__()
        # 3个参数分别是in_channels，out_channels，kernel_size，还可以加padding
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(2704, 512)
        self.fc2 = nn.Linear(512, 84)
        self.fc3 = nn.Linear(84, 3755)  # 命令行参数，后面解释

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 2704)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 初始化模型
model = NetSmall()

# 初始化优化器
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

ch_path = 'checkpoint_3_epoch.pkl'
checkpoint = torch.load(ch_path)
model.load_state_dict(checkpoint['model_state_dict'])

# 打印模型的 state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

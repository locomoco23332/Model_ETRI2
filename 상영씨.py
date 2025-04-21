import torch
import torch.nn as nn

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)), # width 축 절반 (2048 -> 1024)

            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)), # (1024 -> 512)

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)), # (512 -> 256)
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((35, 35)), # 출력 shape을 (35,35)로 맞춤
        )

        self.final_conv = nn.Conv2d(128, 1, kernel_size=(1,1))

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.final_conv(x)
        return x
class GNNLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        
        # bias를 포함하지 않는 basic linear transformation
        self.linear = nn.Linear(dim_in, dim_out, bias=False)
 
    def forward(self, x, adjacency):
        
        # (1) Linear trasnformation
        x = self.linear(x)
        
        # (2) Multiplication with the adjacency matrix A
        x = torch.sparse.mm(adjacency, x)
        return x
#여기서 입맛에 맟춰서 코딩하면됨
# 입력 데이터 예시
input_tensor = torch.randn(1, 1, 35, 2048)

# 모델 생성 및 실행
model = CNNNet()
output = model(input_tensor)
conv = nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=(35, 3),
    stride=(58, 3),
    padding=(0, 0)
)

output=output.permute(3,2,1,0)
#or output=output.unsqueeze(0) output=output.unsqueeze(0) output=conv(output) output=output.squeeze(0),output=output.squeeze(0)

output=output.squeeze(-1)
output=output.squeeze(-1)
print("입력 shape:", input_tensor.shape)
print("출력 shape:", output.shape)

model2=GNNLayer(input,output)

model2(input_data,output)

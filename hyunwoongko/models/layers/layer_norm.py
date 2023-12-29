import torch
from torch import nn

class LayerNorm(nn.Module):
    """
    Layer Normalization은 신경망의 각 layer에서 input의 평균과 표준편차를 계산
    하고, 이를 사용해 input을 정규화하는 기술
    """
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # 여기서 -1은 마지막 dimension을 의미함

        # Layer Normalization 계산. 수식 그대로의 계산임
        # input tensor에서 평균 빼고, 표준편차에 epsilon을 더한 값으로 나눈 뒤 scale 해서 beta를 더해줌
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
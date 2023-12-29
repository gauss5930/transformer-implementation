"""
Transformer의 positional encoding 구현

Transformer에서는 시퀀스 내에서 토큰의 순서를 파악하기 위해 sin & cos 함수를 
활용해 positional encdoing을 진행함
"""

import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()
        
        # input matrix에 더하기 위해 input matrix와 똑같은 사이즈로 만듦
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False # positional encoding에 대해서 기울기를 계산할 필요는 없으므로 False로 둠

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1차원 -> 2차원으로 확장해서 단어의 위치를 표현할 수 있게 해줌

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 여기서 i는 d_model의 인덱스를 의미함

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model))) # 0, 2, 4, ...
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model))) # 1, 3, 5, ...

    def forward(self, x):
        batch_size, seq_len = x.size() # 입력 텐서 x의 형태를 반환함.
        # x's row: batch_size / x's column: seq_len
        # [batch_size: 128, seq_len: 30]

        return self.encoding[:seq_len, :]
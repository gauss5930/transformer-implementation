"""
Multi-Head Attention에 사용되는 scaled dot-product attention
"""

import math

from torch import nn

class ScaleDotProductAttention(nn.Module):
    """
    Query: 우리가 집중할 문장 (decoder)
    Key: Query와의 관계를 확인할 모든 문장 (encoder)
    Value: Key와 똑같은 모든 문장 (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        batch_size, head, length, d_tensor = k.size()

        # 1. dot_product(Q, K^T) / sqrt(d_k)
        k_t = k.transpose(2, 3) # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)

        # 2. masking 적용
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e-9)
        
        # 3. softmax를 적용
        score = self.softmax(score)

        # 4. Value와 dot-product
        v = score @ v

        return v, score
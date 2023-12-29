"""
Transformer의 Multi-Head Attention 구현

Multi-Head Attention에 사요되는 scaled dot-product attention 먼저 구현해보도록 하자.
"""

from torch import nn

from models.layers.scale_dot_product_attention import ScaleDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. weight matrix와 dot-product
        q, k, v = self.w_q(q), self.w_k(k). self.w_v(v)

        # 2. head의 수로 tensor를 나누기
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. scale dot-product attention 진행
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat해서 FC layer로 보내기
        out = self.concat(out)
        out = self.w_concat(out)

        return out
    
    def split(self, tensor):
        """
        head의 수에 따라 tensor를 분할
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

        return tensor
    
    def concat(self, tensor):
        """
        self.split의 역연산
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
from torch import nn

class PositionwiseFeedForward(nn.Module):
    """
    (d_embed X d_ff), (d_ff X d_embed)

    수식: FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
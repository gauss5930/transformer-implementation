from torch import nn

class TokenEmbedding(nn.Embedding):
    """
    torch.nn을 사용하는 token embedding
    가중치 행렬을 사용하는 단어의 representation
    """
    def __init__(self, vocab_size, d_model):
        """
        positional information을 포함하는 token embedding
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)
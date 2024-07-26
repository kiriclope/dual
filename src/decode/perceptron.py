class Perceptron(nn.Module):
    def __init__(self, num_features, dropout_rate=0.0):
        super(Perceptron, self).__init__()
        self.linear = nn.Linear(num_features, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(x)
        hidden = self.linear(x)
        return hidden

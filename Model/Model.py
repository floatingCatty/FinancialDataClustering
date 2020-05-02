import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

class model(nn.Module):
    def __init__(self, d_input, d_model, d_middle, d_output, n_class):
        super(model, self).__init__()
        self.n_class = n_class

        self.d_model = d_model

        self.Encoding = nn.Sequential(
            # embedding
            nn.Linear(d_input, d_model),

            # encoding
            nn.Linear(d_model, d_middle),
            nn.ReLU(),

            nn.Linear(d_middle, d_middle),
            nn.ReLU(),

            nn.Linear(d_middle, d_middle),
            nn.ReLU(),

            nn.Linear(d_middle, d_output),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_output, d_output),
            nn.Sigmoid(),

            nn.Linear(d_output, 2 * d_output),
            nn.Sigmoid(),

            nn.Linear(2 * d_output, d_output),
            nn.Sigmoid(),

            nn.Linear(d_output, self.n_class)
        )

    def forward(self, input):
        N, C, L, D= input.size()

        output = self.Encoding(input)

        output = self.classifier(output)

        output = output.view(N, C, L, -1)

        return output

if __name__ == '__main__':
    data = torch.rand(1000,1,1,200).cuda()
    model = model(
        d_input=200,
        d_model=300,
        d_middle=600,
        d_output=20,
        n_class=10
    ).cuda()
    print(model(data))

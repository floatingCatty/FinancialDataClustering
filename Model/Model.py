import torch.nn as nn
from dataProcessing.dataset import dataset
from torch.utils.data import DataLoader

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

            nn.Linear(d_output, self.n_class),
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        N, L, D = input.size()

        output = self.Encoding(input)

        encoding = output

        output = self.classifier(output)

        output = output.view(N, L, -1)

        return encoding, output

if __name__ == '__main__':
    data = dataset(
        dataPath="E:\Mine\education\\University\contest\\fuwu\data\completion1.csv",
        K=10,
        bsz=1000
    )
    model = model(
        d_input=34,
        d_model=300,
        d_middle=600,
        d_output=20,
        n_class=10
    ).cuda()

    # for i in range(len(data)):
    #     print(data[i]['sample'].view(-1).size())
    loader = DataLoader(dataset=data, batch_size=10, shuffle=True)
    for sample in loader:
        encoding, label = model(sample['sample'].cuda())
        print(encoding.size())
        print(label.size())



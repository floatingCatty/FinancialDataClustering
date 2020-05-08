from torch.utils.data import Dataset
from dataProcessing.utils import getEntnameList
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans

class dataset(Dataset):
    def __init__(self, dataPath, K, bsz):
        self.dataPath = dataPath
        self.data = pd.read_csv(dataPath, encoding='ISO-8859-1')
        self.mbk = MiniBatchKMeans(
            init='k-means++', n_clusters=K, batch_size=bsz,
            n_init=10, max_no_improvement=10, verbose=0
        )

        self.mbk.fit(self.data.drop(columns='entname').to_numpy())

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        assert item < self.__len__()
        sample = self.data[item:item+1]
        sample = sample.drop(['entname'],axis=1)
        sample = sample.to_numpy(dtype='float32')
        if sample.shape == (2,34):
            print(self.nameList[item])
        return {
            'sample':torch.tensor(sample[0], dtype=torch.float32).view(1,-1),
            'label':torch.tensor(self.mbk.predict(sample.reshape(1,-1))).long()
        }



if __name__ == '__main__':
    dataset = dataset(
        dataPath="E:\Mine\education\\University\contest\\fuwu\data\completion1.csv",
        K=10,
        bsz=1000
    )
    sample = dataset[100]['sample'].view(1,1,1,-1).cuda()
    print(sample)
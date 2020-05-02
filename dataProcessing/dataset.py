from torch.utils.data import DataLoader, Dataset
from dataProcessing.utils import getEntnameList
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans

class dataset(Dataset):
    def __init__(self, entnamePath, dataPath, K, bsz):
        self.nameList = getEntnameList(entnamePath)
        self.dataPath = dataPath
        self.data = pd.read_csv(dataPath, encoding='ISO-8859-1')
        self.mbk = MiniBatchKMeans(
            init='k-means++', n_clusters=K, batch_size=bsz,
            n_init=10, max_no_improvement=10, verbose=0
        )
        self.mbk.fit(self.data.to_numpy())

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        assert item < self.__len__()
        sample = self.data[self.data['entname']==self.nameList[item]]
        sample = sample.drop(['entname'],axis=1)
        sample = sample.to_numpy(dtype='float64')

        return (torch.tensor(sample[0]).view(1,-1), self.mbk.predict(sample.reshape(1,-1)))



if __name__ == '__main__':
    dataset = dataset(entnamePath="E:\Mine\education\\University\contest\\fuwu\\trainingSetM\entname.txt",
                      dataPath="E:\Mine\education\\University\contest\\fuwu\\trainingSetM\main_table.csv")
    a = int(input())
    print(dataset[a])
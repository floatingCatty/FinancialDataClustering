import torch
import torch.nn.functional as F
import torch.optim as optim
from dataProcessing.dataset import dataset
from torch.utils.data import DataLoader
from Model.Model import model
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

def cos_matrix_loss(input, target):
    dis_i = F.pdist(input, p=2)
    dis_t = F.pdist(target, p=2)

    loss = F.mse_loss(dis_i, dis_t)

    return loss

def get_loss(tgt_enc, tgt_lab, pre_enc, pre_lab):
    '''
    loss should be composed of two parts:
     first one measures the encoding performance, ensuring relatively equal distance as origin data
     second one constrain the encoding result can be well classified, using cross_entropy
    '''
    N, _, _ = pre_lab.size()

    embedding_loss = cos_matrix_loss(input=pre_enc.view(N,-1), target=tgt_enc.view(N,-1))
    classification_loss = F.cross_entropy(input=pre_lab.view(N,-1), target=tgt_lab.view(N))


    loss = embedding_loss + classification_loss

    return loss


def training_epoch(data, model, optimizer):
    loss_sum = 0
    count = 0
    for item in data:
        count += 1
        enc, lab = model(item["sample"].cuda())
        optimizer.zero_grad()
        loss = get_loss(tgt_enc=item['sample'].cuda(), tgt_lab=item['label'].cuda(), pre_enc=enc, pre_lab=lab)
        loss_sum = loss_sum + loss.item()

        loss.backward()

        optimizer.step()
        torch.cuda.empty_cache()

    ave_loss = loss_sum / count

    return ave_loss

def train(
        dataPath,
        checkPointPath,
        savePath,
        continuePath,
        d_input,
        d_model,
        d_middle,
        d_output,
        n_class,
        cluster_bsz,
        bsz,
        epoch_size,
        lr,
        checkPointRound=10,
        lr_decay=False,
        visualize=False,
        continued=False,
        save=True
):
    net = model(
        d_input=d_input,
        d_model=d_model,
        d_middle=d_middle,
        d_output=d_output,
        n_class=n_class
    )

    data = dataset(
        dataPath=dataPath,
        K=n_class,
        bsz=cluster_bsz
    )

    sample = DataLoader(dataset=data, shuffle=True, batch_size=bsz)

    optimizer = optim.Adam(net.parameters(), lr=lr)

    epoch = 0
    loss_list = []

    if continued:
        checkpoint = torch.load(continuePath)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        epoch = checkpoint['epoch']
        loss_list = checkpoint["loss_list"]

    net = net.cuda().train()

    rest = epoch_size - epoch
    if rest>0:
        for i in tqdm(range(rest)):
            epoch += 1

            if lr_decay:
                new_lr = lr / math.log2(epoch + 1)
                for param in optimizer.param_groups:
                    param['lr'] = new_lr
            else:
              for param in optimizer.param_groups:
                param['lr'] = lr

            ave_loss = training_epoch(
                data=sample,
                model=net,
                optimizer=optimizer
            )
            loss_list.append(ave_loss)
            print(ave_loss)

            if epoch % checkPointRound == 0:
                torch.save(obj={
                    'model_state_dict':model.state_dict(),
                    'epoch':epoch,
                    'optimizer_state_dict':optimizer.state_dict(),
                    'loss_list':loss_list
                }, f=checkPointPath+"\\"+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+".pth")
    else:
        print("Current training epoch has already finished.")

    if save:
        torch.save(model, savePath + "\\"+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+".pth")

    if visualize:
        plt.plot(loss_list)
        plt.show()

    return True


if __name__ == '__main__':
    root = "E:\Mine\education\\University\contest\\fuwu\project_test"

    dataPath = "E:\Mine\education\\University\contest\\fuwu\data\completion1.csv"
    checkPointPath = root + "\\" + "checkpoint"
    savePath = root + "\\" + "save"
    continuePath = root + "\\" + ""

    train(
        dataPath=dataPath,
        checkPointPath=checkPointPath,
        savePath=savePath,
        continuePath=continuePath,
        d_input=34,
        d_model=300,
        d_middle=600,
        d_output=20,
        n_class=10,
        cluster_bsz=1000,
        bsz=100,
        epoch_size=100,
        lr=1e-4,
        checkPointRound=10,
        lr_decay=False,
        visualize=True,
        continued=False,
        save=True
    )



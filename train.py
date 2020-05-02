import torch
import torch.nn.functional as F
import torch.optim as optim
from dataProcessing.dataset import dataset
from Model.Model import model
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

def get_loss(tgt, pre):
    N, _, _ = pre.size()
    loss = F.mse_loss(pre, tgt, reduction="mean")
    return loss / N


def training_epoch(data, model, optimizer):
    loss_sum = 0
    count = 0
    for item in data:
        count += 1
        pre = model(item["sequence"].cuda())
        optimizer.zero_grad()
        # loss = get_loss(tgt=item["position"].cuda(), pre=pre)
        loss = get_loss()
        loss_sum = loss_sum + loss.item()

        loss.backward()

        optimizer.step()
        torch.cuda.empty_cache()

    ave_loss = loss_sum / count

    return ave_loss

def train(
        entnamePath,
        dataPath,
        n_class,
        cluster_bsz,
        checkPointPath,
        savePath,
        continuePath,
        epoch_size,
        lr,
        lr_decay=False,
        visualize=False,
        continued=False,
        save=True
):
    model = model(

    )

    sample = dataset(entnamePath, dataPath, K=n_class, bsz=cluster_bsz)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch = 0
    loss_list = []

    if continued:
        checkpoint = torch.load(continuePath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        epoch = checkpoint['epoch']
        loss_list = checkpoint["loss_list"]

    model = model.cuda().train()

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
                model=model,
                optimizer=optimizer
            )
            loss_list.append(ave_loss)
            print(ave_loss)

            # if epoch % 2 == 0:
            #     torch.save(obj={
            #         'model_state_dict':model.state_dict(),
            #         'epoch':epoch,
            #         'optimizer_state_dict':optimizer.state_dict(),
            #         'loss_list':loss_list
            #     }, f=checkPointPath+"\\"+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+".pth")
    else:
        print("Current training epoch has already finished.")

    if save:
        torch.save(model, savePath + "\\"+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+".pth")

    if visualize:
        plt.plot(loss_list)
        plt.show()

    return True


if __name__ == '__main__':

    checkPointPath = "E:\Mine\education\科研\project\HKtec\DBS(GPU)\checkPoint"
    savePath = "E:\Mine\education\科研\project\HKtec\DBS(GPU)\save"

    continuePath = "E:\Mine\education\科研\project\HKtec\DBS(GPU)\checkPoint\\2020-04-27 09_19_44.pth"

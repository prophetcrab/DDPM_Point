import os

import torch
import torch.distributed as dist
from torch.utils.data import dataloader
from tqdm import tqdm

from Utils.utils import *


def fit_one_epoch(
        diffusion_model, diffusion_model_train, optimizer, loss_History, dataloader,
        local_rank, epoch_step, epoch, Epoch, save_period, save_dir,
        cuda,
        fp16, scaler,
):
    total_loss = 0
    """
    diffusion_model : 用来更新的diffusion_model
    diffusion_model_train : 用来训练的diffusion_model
    optimizer : 优化器
    loss_History : 
    epoch_step : 参数指定的进度条总步数，每个epoch需要执行的步骤数量
    epoch : 当前epoch
    Epoch : 总Epoch
    save_period : 每隔save_period个世代就保存一下
    save_dir:保存的路径
    Local_rank : 用于GPU索引和分布式训练
    cuda:是否在cuda上训练
    fp16：是否使用fp16精度
    scaler: 用于fp16精度训练
    """
    if local_rank == 0:
        print('start train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    for iteration, point_cloud in enumerate(dataloader):
        if iteration >= epoch_step:
            break

        with torch.no_grad():
            if cuda:
                point_cloud = point_cloud.cuda(local_rank)

        if not fp16:
            optimizer.zero_grad()
            diffusion_loss = torch.mean(diffusion_model_train(point_cloud))
            diffusion_loss.backward()
            optimizer.step()
        else:
            from  torch.cuda.amp import autocast
            optimizer.zero_grad()
            with autocast():
                diffusion_loss = torch.mean(diffusion_model_train(point_cloud))
            #反向传播
            scaler.scale(diffusion_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        diffusion_model.update_ema()

        total_loss += diffusion_loss.item()
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    total_loss = total_loss / epoch_step

    if local_rank == 0:
        pbar.close()
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.4f ' % (total_loss))
        loss_History.append_loss(epoch + 1, total_loss=total_loss)

        #--------------------------------------#
        # 是否每个epoch之后都sample一次
        #--------------------------------------#

        print('Show_result : ')
        show_result(epoch + 1, diffusion_model, point_cloud.device)
        #------------------------------------#
        #    每隔若干个世代保存一次
        #------------------------------------#
        if (epoch + 1) % save_period == 0 or (epoch + 1) == Epoch:
            torch.save(diffusion_model.state_dict(),
                       os.path.join(save_dir, "Diffusion_Epoch%d-GLoss%.4f.pth" % (epoch + 1, total_loss)))

        torch.save(diffusion_model.state_dict(), os.path.join(save_dir, "diffusion_model_last_epoch_weights.pth"))




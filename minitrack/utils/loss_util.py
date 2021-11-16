import torch
import math
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from .dataset import yolo_dataset_collate
from tqdm import tqdm
import json
import time

def box_ciou(b1, b2, device):
    """
    输入为：
    b1:tensor,shape=(B*levels*H*W*A,4),xywh
    b2:tensor,shape=(B*levels*H*W*A,4),xywh
    返回为：
    ciou矩阵:tensor,shape=(B*levels*H*W*A)一维
    """
    # 求出预测框左上角右下角
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    # 求出真实框左上角右下角
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # 求真实框和预测框所有的iou
    intersect_mins = torch.max(b1_mins, b2_mins)
    intersect_maxes = torch.min(b1_maxes, b2_maxes)
    intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes, device=device))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / torch.clamp(union_area, min=1e-6)

    # 计算中心的差距
    center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), dim=-1)

    # 找到包裹两个框的最小框的左上角和右下角
    enclose_mins = torch.min(b1_mins, b2_mins)
    enclose_maxes = torch.max(b1_maxes, b2_maxes)
    enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes, device=device))
    # 计算对角线距离
    enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), dim=-1)
    ciou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal, min=1e-6)

    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(b1_wh[..., 0] / torch.clamp(b1_wh[..., 1], min=1e-6)) - torch.atan(
        b2_wh[..., 0] / torch.clamp(b2_wh[..., 1], min=1e-6))), 2)
    alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)
    ciou = ciou - alpha * v
    return ciou.squeeze()


def BCELoss(pred, target):
    epsilon = 1e-7
    pred = torch.clamp(pred, min=epsilon, max=1.0 - epsilon)  # 避免log(0)为nan
    output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    return output


def FocalLoss(pred, target, gamma=2.0, alpha=0.25):
    epsilon = 1e-7
    pred = torch.clamp(pred, min=epsilon, max=1.0 - epsilon)  # 避免log(0)为nan
    output = -alpha * torch.pow((1.0 - pred), gamma) * target * torch.log(pred) - \
             (1.0 - alpha) * torch.pow(pred, gamma) * (1.0 - target) * torch.log(1.0 - pred)
    return output


def LogSoftmaxLoss(pred, target):
    epsilon = 1e-7
    pred = torch.clamp(pred, min=epsilon)  # 避免log(0)为nan
    output = -target * torch.log(pred)
    return output

def get_smooth_labels(labels, label_smoothing, num_classes):
    return labels * (1.0 - label_smoothing) + label_smoothing / num_classes


def fit_ont_epoch(net,netparal, optimizer,lr_scheduler,Loss, epoch, iteration_size, iteration_size_val, gen, genval, Epoch, device):
    # gen,genval为dataloader
    total_loss = 0  # 记录一个epoch的loss和
    val_loss = 0
    start_time = time.time()

    with tqdm(total=iteration_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= iteration_size:
                break
            images, targets = batch[0], batch[1]

            images = images.to(device)

            optimizer.zero_grad()
            outputs = netparal(images)

            loss = Loss(outputs, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            waste_time = time.time() - start_time

            pbar.set_postfix(**{'loss': total_loss / (iteration + 1),'step/s': waste_time})
            pbar.update(1)

            start_time = time.time()

    print('Start Validation')
    netparal.eval()
    with tqdm(total=iteration_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= iteration_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():  # 验证时不要计算梯度

                images_val = images_val.to(device)

                optimizer.zero_grad()
                outputs = netparal(images_val)

                loss = Loss(outputs, targets_val)
                val_loss += loss.item()
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (iteration_size + 1), val_loss / (iteration_size_val + 1)))

    print('Saving state, iter:', str(epoch + 1))
    checkpoint = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1,'scheduler': lr_scheduler.state_dict()}  # epoch保存的是下一次训练的次数，所以加1
    # 注意这里保存的是net。net是DataParallel，如果保存net会在开头加module。干脆保存model，不会在开头加module
    torch.save(checkpoint, 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % ((epoch + 1), total_loss / (iteration_size + 1), val_loss / (iteration_size_val + 1)))

def train_funct(net,start_epoch,device,cfg_path,loss,train_dataset,val_dataset):
    cfg=json.load(open(cfg_path))
    Freeze_Epoch = cfg['Freeze_Epoch']
    Unfreeze_Epoch = cfg['Unfreeze_Epoch']
    Freeze_lr=cfg['Freeze_lr']
    Unfreeze_lr=cfg['Unfreeze_lr']
    Batch_size_freeze = cfg['Batch_size_freeze']
    Batch_size_nofreeze = cfg['Batch_size_nofreeze']

    Cosine_lr = cfg['Cosine_lr']

    if cfg['cuda']:
        netparal = torch.nn.DataParallel(net)
        # 多GPU分发
        # 优化选择最合适的卷积方式，不同卷积核、输入的最合适卷积方式不同，设置为True，会再开始前选择最合适的
        # 当然训练的输入的大小不变时，适合设置为True，若一直变化，反而会消耗很多时间去搜索最优
        cudnn.benchmark = True

    else:
        netparal=net

    netparal = netparal.to(device)
    if start_epoch< Freeze_Epoch:
        optimizer = optim.Adam([{'params':netparal.parameters()},{'params':loss.parameters()}], Freeze_lr, weight_decay=5e-4)  # L2正则化

        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

        if start_epoch != 0:
            checkpoint = torch.load(cfg['torch_model_path'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
            del checkpoint

        gen = DataLoader(train_dataset, shuffle=False, batch_size=Batch_size_freeze,drop_last=True, collate_fn=yolo_dataset_collate,pin_memory=True)
        gen_val = DataLoader(val_dataset, shuffle=False, batch_size=Batch_size_freeze, drop_last=True, collate_fn=yolo_dataset_collate,pin_memory=True)

        iteration_size = max(1, len(train_dataset) // Batch_size_freeze)  # 至少为1
        iteration_size_val =max(1, len(val_dataset) // Batch_size_freeze)
        #   冻结backbone
        for param in net.backbone.parameters():
            param.requires_grad = False

        while (start_epoch < Freeze_Epoch):
            netparal.train()

            for module in net.backbone.modules(): # backbone的BN冻结
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()

            fit_ont_epoch(net,netparal,optimizer,lr_scheduler, loss, start_epoch, iteration_size, iteration_size_val, gen, gen_val,Freeze_Epoch,device)
            lr_scheduler.step()
            start_epoch += 1  # 加1
            train_dataset.shuffle()

    if Freeze_Epoch <= start_epoch and start_epoch< Unfreeze_Epoch:

        optimizer = optim.Adam([{'params':netparal.parameters()},{'params':loss.parameters()}], Unfreeze_lr, weight_decay=5e-4)  # L2正则化

        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

        if start_epoch != Freeze_Epoch:
            checkpoint = torch.load(cfg['torch_model_path'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
            del checkpoint

        gen = DataLoader(train_dataset, shuffle=False, batch_size=Batch_size_nofreeze, pin_memory=True,drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=False, batch_size=Batch_size_nofreeze, pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)

        iteration_size = max(1, len(train_dataset) // Batch_size_nofreeze)  # 至少为1
        iteration_size_val =max(1, len(val_dataset) // Batch_size_nofreeze)
        #   解冻
        for param in net.backbone.parameters():
            param.requires_grad = True

        while (start_epoch < Unfreeze_Epoch):
            netparal.train()
            fit_ont_epoch(net,netparal,optimizer,lr_scheduler,loss,start_epoch, iteration_size, iteration_size_val, gen, gen_val, Unfreeze_Epoch,device)
            lr_scheduler.step()
            start_epoch += 1
            train_dataset.shuffle()
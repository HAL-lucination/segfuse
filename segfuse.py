from __future__ import print_function
import os
import sys
sys.path.append('../../')
import argparse
import yaml
import time
import tqdm
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import Utils
import Utils.Metrics as metrics
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from termcolor import colored
from itertools import chain
from dataset.structured3D import OmniDepthDataset
from Utils.visualization import VisdomVisualizer
from Utils.berhu import Berhu as ReverseHuberLoss
from Utils.Segmentation import get_seg as get_seg
import os.path as osp 

parser = argparse.ArgumentParser(description='Training script for 360 layout',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mode', default='test', type=str, help='train/test mode')
parser.add_argument('--pre', default=None, type=int, help='pretrain(default: latest)')
parser.add_argument('--log', default='Results.txt', type=str, help='log file name')
parser.add_argument('--gpu_id', required=True, type=str, help='gpu id')
args = parser.parse_args()

network_type = 'RectNet'  # 'RectNet' or 'UResNet'
experiment_name = 'Segfuse'
root_path = ''
train_file_list = ''  # File with list of training files
val_file_list = ''  # File with list of validation files
max_steps_file = ''  # File with number of max_steips for progress show
checkpoint_dir = osp.join('experiments', experiment_name)
checkpoint_path = None
load_weights_only = False
batch_size = 1
num_workers = 8
lr = 3e-4
step_size = 3
lr_decay = 0.5
num_epochs = 20
validation_freq = 1
visualization_freq = 1
validation_sample_freq = -1
device_ids = [0]
num_samples = 0

# avg_meters

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
torch.backends.cudnn.deterministic = True


def train(args, config, dataset_train, dataset_val, model, saver, writer):
    vis = VisdomVisualizer("Bifuse_real")
    [_, offset] = saver.LoadLatestModel(model, args.pre)
    finish_epoch = offset
    offset = offset * len(dataset_train)

    if config["training_stage"] ==   1:
        #param = chain(model.conv_e2c.parameters(), model.conv_c2e.parameters(), model.conv_mask.parameters())
        param = chain(model.conv_e2c.parameters(), model.conv_c2e.parameters())
    elif config["training_stage"] == 2:
        param = chain(model.conv_mask.parameters(), model.conv_e2c.parameters(), model.conv_c2e.parameters())
    elif config["training_stage"] == 3:
        param = model.parameters()
    else:
        print("training_stage must be 1 or 2 or 3")
    optim = torch.optim.Adam(param, lr=config['lr'])
    start_epoch = 0

    for epoch in range(finish_epoch, config['epochs']):
    	# update learning rate
        offset = train_an_epoch(config, model, dataset_train, optim, writer, epoch, offset, vis)
        saver.Save(model, epoch)

        global_step = (epoch + 1) * len(dataset_train)
        if dataset_val is not None:
            val_results = val_an_epoch(dataset_val, model, config, writer, vis)
            print(colored('\nDense Results: ', 'magenta'))
            for name, val in val_results['dense-equi'].items():
                print(colored('- {}: {}'.format(name, val), 'magenta'))
                writer.add_scalar('C-val-dense-metric/{}'.format(name), val, epoch)
            with open(args.log, 'a') as f:
                f.write('This is %d epoch:\n'%(epoch))
                for name, val in val_results['dense-equi'].items():
                    f.write('--- %s: %f\n'%(name, val))
                f.close()

def train_an_epoch(config, model, loader, optim, writer, epoch, step_offset, vis : VisdomVisualizer):
    model.train()
    meters = metrics.Metrics(['rmse', 'mae', 'mre'])
    threshold = 10
    berhu = ReverseHuberLoss(threshold=threshold, reduction="mean")
    iter_time = 0
    loss = 0
    CE = Utils.CETransform()
    grid = Utils.Equirec2Cube(None, 512//2, 1024//2, 256//2, 90).GetGrid()
    d2p = Utils.Depth2Points(grid)

    i = 0
    for data in tqdm.tqdm(loader):
        i += 1
        it = i + step_offset
        #raw_rgb_var, rgb_var, depth_var = data['raw_rgb'], data['rgb'], data['depth']
        raw_rgb_var, rgb_var = data[0], data[0]
        depth_var = data[1]

        raw_rgb_var = raw_rgb_var.cuda()
        rgb_var = rgb_var.cuda()
        depth_var = depth_var.cuda()

        valid = (depth_var <= threshold).float() * (depth_var != 0).float()
        rgb_equi = rgb_var

        pred_var, pred_cube_var, refine, pred_seg = model(rgb_equi)
        #'''
        cube_pts = d2p(pred_cube_var)
        pred_cube_var = CE.C2E(torch.norm(cube_pts, p=2, dim=3).unsqueeze(1))
        #'''
        loss_var_dict = dict()
        loss_var_dict['BerHu-equi'] = berhu(pred_var, depth_var, valid)
        loss_var_dict['BerHu-cube'] = berhu(pred_cube_var, depth_var, valid)

        # added seg loss NLLLoss()/CrossEntropyLoss()
        gt_seg = get_seg(raw_rgb_var).squeeze(1).to(dtype=torch.long)
        loss_seg = nn.CrossEntropyLoss() # negative log likelihood loss
        loss_var_dict['NegLog-seg'] = loss_seg(pred_seg.cuda(), gt_seg.cuda())

        # SegFuse
        total_loss_var = loss_var_dict['BerHu-equi'] + loss_var_dict['BerHu-cube'] + loss_var_dict['NegLog-seg']
        # BiFuse
        # total_loss_var = loss_var_dict['BerHu-equi'] + loss_var_dict['BerHu-cube']

        optim.zero_grad()
        total_loss_var.backward()
        optim.step()

        if it % config['print_step'] == 0:
            vis.append_loss(epoch,it, loss_var_dict['BerHu-equi'].detach(), 'BerHu-equi')
            vis.append_loss(epoch,it, loss_var_dict['BerHu-cube'].detach(), 'BerHu-cube')
            vis.append_loss(epoch,it, loss_var_dict['NegLog-seg'].detach(), 'NegLog-seg')
            if config["training_stage"] != 1:
                vis.append_loss(epoch,it, total_loss_var.detach(), 'total_loss_var')

            vis.show_images(raw_rgb_var, "raw_rgb_var")
            vis.show_map(pred_var.detach().cpu(), "pred_equi")
            vis.show_map(pred_cube_var.detach().cpu(), "pred_cube")
            vis.show_map(depth_var.detach().cpu(), "depth")
            if refine is not None:
                vis.show_map(refine.detach().cpu(), "refine")

    return it

def val_an_epoch(loader, model, config, writer, vis : VisdomVisualizer):
    model = model.eval()
    meters = metrics.Metrics(config['metrics'])
    avg_meters = metrics.MovingAverageEstimator(config['metrics'])
    avg_meters_cube = metrics.MovingAverageEstimator(config['metrics'])

    pbar = tqdm.tqdm(loader)
    pbar.set_description('Validation process')
    #pbar = loader
    gpu_num = torch.cuda.device_count()

    CE = Utils.CETransform()
    grid = Utils.Equirec2Cube(None, 512, 1024, 256, 90).GetGrid()
    d2p = Utils.Depth2Points(grid)

    with torch.no_grad():
        for it, data in enumerate(pbar):
            # raw_rgb_var, rgb_var, depth_var = data['raw_rgb'], data['rgb'], data['depth']
            raw_rgb_var, rgb_var = data[0], data[0]
            depth_var = data[1]
            raw_rgb_var, rgb_var, depth_var = raw_rgb_var.cuda(), rgb_var.cuda(), depth_var.cuda()

            inputs = rgb_var
            if inputs.shape[0] % gpu_num == 0:
                raw_pred_var, pred_cube_var, *other = model(inputs)
            else:
                raw_pred_var = []
                pred_cube_var = []
                count = inputs.shape[0] // gpu_num
                lf = inputs.shape[0] % gpu_num
                for gg in range(count):
                    a = inputs[gg*gpu_num:(gg+1)*gpu_num]
                    a, b = model(a)
                    raw_pred_var.append(a)
                    pred_cube_var.append(b)

                a = inputs[count*gpu_num:]
                a, b = model.module(a)
                raw_pred_var.append(a)
                pred_cube_var.append(b)
                raw_pred_var = torch.cat(raw_pred_var, dim=0)
                pred_cube_var = torch.cat(pred_cube_var, dim=0)
            #'''
            # cube_pts = d2p(pred_cube_var)
            # pred_cube_var = CE.C2E(torch.norm(cube_pts, p=2, dim=3).unsqueeze(1))
            #'''
            for i in range(raw_pred_var.shape[0]):
                pred = raw_pred_var[i:i+1].data.cpu().numpy()
                # pred_cube = pred_cube_var[i:i+1].data.cpu().numpy()
                depth = depth_var[i:i+1].data.cpu().numpy()

                results = meters.compute(pred.clip(0, 10), depth.clip(0, 10))
                # results_cube = meters.compute(pred_cube.clip(0, 10), depth.clip(0, 10))
                avg_meters.update(results)

                # avg_meters_cube.update(results_cube)
                # print (results)
                # print (results_cube)

    # Print final results and log to tensorboard
    final_results = {
        'dense-equi': avg_meters.compute(),
    }
    print(final_results)
    return final_results

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
        print (json.dumps(config, indent=4))

    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    prediction_size = (256,512)

    dataset_train = torch.utils.data.DataLoader(
        dataset=OmniDepthDataset(
        root_path = root_path,
        path_to_img_list=train_file_list),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
            )

    dataset_val = torch.utils.data.DataLoader(
        dataset=OmniDepthDataset(
        root_path = root_path,
        path_to_img_list=val_file_list),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
            )

    saver = Utils.ModelSaver(config['save_path'])
    from models.FCRN import MyModel as ResNet
    model = ResNet(
    		layers=config['model_layer'],
    		decoder=config['decoder_type'],
    		output_size=None,
    		in_channels=3,
    		pretrained=True,
            prediction_size=prediction_size,
            training_stage=config["training_stage"]
    		).cuda()
    if args.mode == 'train':
        #writer = Utils.visualizer(config['exp_path'])
        writer = SummaryWriter(config['exp_path'])
        train(args, config, dataset_train, dataset_val, model, saver, writer)
    else:
        saver.LoadLatestModel(model, args.pre)

        writer = None
        model = nn.DataParallel(model)
        results = val_an_epoch(dataset_val, model, config, writer)
        print(colored('\nDense Results: ', 'magenta'))
        for name, val in results['dense-equi'].items():
            print(colored('- {}: {}'.format(name, val), 'magenta'))
        for name, val in results['dense-cube'].items():
            print(colored('- {}: {}'.format(name, val), 'magenta'))


if __name__ == '__main__':
    main()

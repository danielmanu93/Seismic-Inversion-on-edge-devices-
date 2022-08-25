import datetime
import os
import sys
import time
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import SequentialSampler
import numpy as np
import utils
import network
from vis import plot_kimb_velocity, plot_kimb_seismic, plot_kimb_trace, plot_kimb_profile
from dataset import FWIDataset
import transforms as T
import pytorch_ssim

def evaluate(model, criterions, dataloader, device, 
                data_min, data_max, label_min, label_max,
                vis=False, vis_path=None, vis_batch=0, vis_sample=0):
    model.eval()
    header = 'Test:'
    label_list, label_pred_list= [], [] # store denormalized predcition & gt in numpy 
    label_tensor, pred_tensor = [], [] # store normalized prediction & gt in tensor
    if vis:
        vis_vel, vis_pred = [], []
    with torch.no_grad():
        batch_idx = 0
        # for data, label in dataloader:
        for i in range(1):
            data, label = iter(dataloader).next()
            data = data.type(torch.FloatTensor).to(device, non_blocking=True)
            label = label.type(torch.FloatTensor).to(device, non_blocking=True)
            
            label_np = T.tonumpy_denormalize(label, label_min, label_max, exp=False)
            label_list.append(label_np)
            label_tensor.append(label)
            
            pred = model(data)
            label_pred_np = T.tonumpy_denormalize(pred, label_min, label_max, exp=False)

            label_pred_list.append(label_pred_np)
            pred_tensor.append(pred)
            
            data_np = data.numpy()

            trace1 = data_np[1, 1, :, 10]
            trace2 = data_np[1, 1, :, 20]
            trace3 = data_np[1, 1, :, 30]
            
            prof1 = label_pred_np[1, 0, :, 10]
            prof2 = label_pred_np[1, 0, :, 20]
            prof3 = label_pred_np[1, 0, :, 30]

            # Visualization
            if vis and batch_idx < vis_batch:
                vis_vel.append(label_np[:vis_sample])
                vis_pred.append(label_pred_np[:vis_sample])

                for i in range(vis_sample):
                    plot_kimb_velocity(label_pred_np[i, 0], label_np[i, 0], f'{vis_path}/PD.png')
                    plot_kimb_seismic(data_np, f'{vis_path}/Seismic.png')
                    plot_kimb_trace(trace1, trace2, trace3, f'{vis_path}/Trace.png')
                    plot_kimb_profile(prof1, prof2, prof3, f'{vis_path}/VelProfile.png')
            batch_idx += 1

    label, label_pred = np.concatenate(label_list), np.concatenate(label_pred_list)
    label_t, pred_t = torch.cat(label_tensor), torch.cat(pred_tensor)
    l1 = nn.L1Loss()
    print(f'MAE: {l1(label_t, pred_t)}')
    l2 = nn.MSELoss()
    print(f'MSE: {l2(label_t, pred_t)}')
    ssim_loss = pytorch_ssim.SSIM(window_size=11)
    print(f'SSIM: {ssim_loss(label_t / 2 + 0.5, pred_t / 2 + 0.5)}') # (-1, 1) to (0, 1)

    if vis:
        vel, vel_pred = np.concatenate(vis_vel), np.concatenate(vis_pred)


def main(args):

    utils.mkdir(args.output_path)
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    if args.dataset == 'salt_down':
        data_min = -20
        data_max = 38
        label_min = 1500
        label_max = 4500

    transform_valid_data = torchvision.transforms.Compose([
        T.LogTransform(k=args.k),
        T.MinMaxNormalize(T.log_transform(data_min, k=args.k), T.log_transform(data_max, k=args.k))
    ])

    transform_valid_label = torchvision.transforms.Compose([
        T.MinMaxNormalize(label_min, label_max)
    ])
    if args.val_anno[-3:] == 'txt':
        dataset_valid = FWIDataset(
            args.val_anno,
            preload=True,
            sample_ratio=args.sample_ratio,
            file_size=args.file_size,
            transform_data=transform_valid_data,
            transform_label=transform_valid_label
        )
    else:
        dataset_valid = torch.load(args.val_anno)

    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=True, collate_fn=default_collate, shuffle=True)

    if args.model not in network.model_dict:
        print('Unsupported model.')
        sys.exit()
     
    if args.up_mode:    
        model = network.model_dict[args.model](upsample_mode=args.up_mode).to(device)
    else:
        model = network.model_dict[args.model]().to(device)

    criterions = {
        'MAE': lambda x, y: np.mean(np.abs(x - y)),
        'MSE': lambda x, y: np.mean((x - y) ** 2)
    }

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        print(checkpoint)
        model.load_state_dict(checkpoint['model'])
        print(model.state_dict)
    
    if args.vis:
        vis_folder = f'visualization_{args.vis_suffix}' if args.vis_suffix else 'visualization'
        vis_path = os.path.join(args.output_path, vis_folder)
        utils.mkdir(vis_path)
    
    start_time = time.time()
    if args.vis:
        evaluate(model, criterions, dataloader_valid, device, data_min, data_max, label_min, label_max,
                vis=True, vis_path=vis_path, vis_batch=args.vis_batch, vis_sample=args.vis_sample)
    else:
        evaluate(model, criterions, dataloader_valid, device, data_min, data_max, label_min, label_max)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testing time {}'.format(total_time_str))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='FCN Testing')
    parser.add_argument('--anno-path', default='relevant_files', help='dataset files location')
    parser.add_argument('-o', '--output-path', default='C:\\Users\\DANIEL\\Desktop\\exp\\fcnvmb\\UPFWI\\checkpoints', help='path where to save')
    parser.add_argument('-v', '--val-anno', default='C:\\Users\\DANIEL\\Desktop\\exp\\fcnvmb\\UPFWI\\relevant_files\\salt_down_valid.txt', help='name of val anno')
    parser.add_argument('-fs', '--file-size', default=10, type=int, help='samples per data file')
    parser.add_argument('-ds', '--dataset', default='salt_down', type=str, help='dataset option for normalization')
    parser.add_argument('-sr', '--sample_ratio', type=int, default=1, help='subsample ratio of data')
    parser.add_argument('-n', '--save-name', default='fcn_kimb_1', help='saved name for this run')
    parser.add_argument('-s', "--suffix", type=str, default=None)
    parser.add_argument('-m', '--model', default="FCN4_1", help='select inverse model')
    parser.add_argument('--up_mode', default=None, help='upsample mode')
    parser.add_argument('-d', '--device', default='cpu', help='device')
    parser.add_argument('-b', '--batch-size', default=10, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers (default: 16)')
    parser.add_argument('--k', default=1, type=float, help='k in log transformation')
    parser.add_argument('--resume', default='C:\\Users\\DANIEL\\Desktop\\exp\\fcnvmb\\UPFWI\\checkpoints\\fcn_kimb_1\\run1\\model_1000.pth', help='resume from checkpoint')
    parser.add_argument('--vis', default=True, help='visualization option', action="store_true")
    parser.add_argument('--vis_suffix', default='1000_test', type=str, help='visualization suffix')
    parser.add_argument('--vis_batch', help='number of batch to be visualized', default=1, type=int)
    parser.add_argument('--vis_sample', help='number of samples in a batch to be visualized', default=1, type=int)

    args = parser.parse_args()

    args.output_path = os.path.join(args.output_path, args.save_name, args.suffix or '')
    args.val_anno = os.path.join(args.anno_path, args.val_anno)

    return args


if __name__ == '__main__':

    args = parse_args()
    main(args)

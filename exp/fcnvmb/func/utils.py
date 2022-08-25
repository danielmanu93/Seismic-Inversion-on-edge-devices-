# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
from math import log10
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F
import matplotlib
matplotlib.use('tkAgg') 
import matplotlib.pyplot as plt
from matplotlib import pyplot
import scipy.io
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.ndimage as ndimage
from ParamConfig import *
import matplotlib.patches as patches
plt.rcParams.update({'figure.max_open_warning': 0})

def turn(GT):
    dim = GT.shape
    for j in range(0,dim[1]):
        for i in range(0,dim[0]//2):
            temp    = GT[i,j]
            GT[i,j] = GT[dim[0]-1-i,j]
            GT[dim[0]-1-i,j] = temp
    return GT 


def PSNR(prediction, target):
    prediction = Variable(torch.from_numpy(prediction))
    target     = Variable(torch.from_numpy(target))
    zero       = torch.zeros_like(target)   
    criterion  = nn.MSELoss(size_average=True)    
    MSE        = criterion (prediction, target)
    total      = criterion (target, zero)
    psnr       = 10. * log10(total.item() / MSE.item())
    return psnr

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window     = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1    = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2    = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    L  = 255
    C1 = (0.01*L) ** 2
    C2 = (0.03*L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)



def SSIM(img1, img2, window_size=11, size_average=True):
    img1 = Variable(torch.from_numpy(img1))
    img2 = Variable(torch.from_numpy(img2))
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def SaveTrainResults(loss,SavePath,font2,font3):
    fig,ax  = plt.subplots()
    plt.plot(loss[1:], linewidth=2)
    plt.plot(loss, linewidth=2)
    ax.set_xlabel('Num. of epochs', font2)
    ax.set_ylabel('MSE Loss', font2)
    ax.set_title('Training', font3)
    ax.set_xlim([1,500])
    for label in ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(12)
    ax.grid(linestyle='dashed',linewidth=0.5)
     
    plt.savefig(SavePath+'TrainLoss', transparent = True)
    data = {}
    data['loss'] = loss
    np.save(SavePath+'TrainLoss.npy',data)
    plt.show()
    plt.close()

def SaveTestResults(TotPSNR,TotSSIM,Prediction,GT,SavePath):
    data = {}
    data['TotPSNR'] = TotPSNR
    data['TotSSIM'] = TotSSIM    
    data['GT']      = GT
    data['Prediction'] = Prediction
    np.save(SavePath+'TestResults.npy',data)
    
# Plot Ground Truth Image  
def PlotGroundTruth(gt,label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath):
    GT = gt.reshape(label_dsp_dim[0],label_dsp_dim[1])
    fig,ax1 = plt.subplots(figsize=(4, 4))    
    im1     = ax1.imshow(GT,extent=[0,label_dsp_dim[1]*label_dsp_blk[1]*dh/1000., \
                              0,label_dsp_dim[0]*label_dsp_blk[0]*dh/1000.],vmin=minvalue,vmax=maxvalue)
    divider = make_axes_locatable(ax1)
    cax1    = divider.append_axes("right",size="5%",pad=0.05)
    plt.colorbar(im1,ax=ax1,cax=cax1).set_label('Velocity (m/s)')
    plt.tick_params(labelsize=5)
    for label in  ax1.get_xticklabels()+ax1.get_yticklabels():
        label.set_fontsize(5)
    ax1.set_xlabel('Position (km)',font2)
    ax1.set_ylabel('Depth (km)',font2)
    ax1.set_title('Ground truth',font3)
    ax1.invert_yaxis()
    plt.subplots_adjust(bottom=0.20,top=0.65,left=0.08,right=0.95)
    plt.savefig(SavePath+'GT',transparent=True)
    plt.close()

# Plot Prediction
def PlotPrediction(pd,vel_rec1, vel_rec2, vel_rec3,label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath):
    PD = pd.reshape(label_dsp_dim[0],label_dsp_dim[1])
    fig2,ax=plt.subplots(figsize=(9, 5))
    im2=ax.imshow(PD,extent=[0,label_dsp_dim[1]*label_dsp_blk[1]*dh/1000., \
                              0,label_dsp_dim[0]*label_dsp_blk[0]*dh/1000.],vmin=minvalue,vmax=maxvalue)
    divider = make_axes_locatable(ax)
    cax    = divider.append_axes("right",size="5%",pad=0.05)
    for label in ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(10)
        # ax2.tick_params(labelsize=6)   
    ax.set_xlabel('Position (km)',fontsize=10)
    ax.set_ylabel('Depth (km)',fontsize=10)
    ax.set_title('Prediction',{'family': 'Times New Roman','weight': 'normal','size': 16})
    ax.invert_yaxis()
    ax.vlines(x=[vel_rec1/100, vel_rec2/100, vel_rec3/100], ymin=0, ymax=2, colors=['r', 'b', 'k'], linestyles='--')
    fig2.patch.set_facecolor("#C0C0C0") ##C5C9C7
    fig2.patch.set_edgecolor("#C0C0C0") ##C5C9C7
    cbar=plt.colorbar(im2,ax=ax,cax=cax)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(label='Velocity (m/s)', size=12)
    # plt.subplots_adjust(bottom=0.20,top=0.65,left=0.08,right=0.95)
    plt.savefig(SavePath+'PD', facecolor=fig2.get_facecolor(), 
                                edgecolor=fig2.get_edgecolor(), transparent=True)

#Display seismic data
def display_seismic(seis, seis_rec1, seis_rec2, seis_rec3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath):
    vmin, vmax = np.min(seis), np.max(seis)
    fig, ax = plt.subplots(figsize=(4, 4))
    seis = ax.imshow(seis, cmap='gray', aspect="auto", extent=[0,data_dsp_dim[1]*data_dsp_blk[1]*dh/1000., \
                            data_dsp_dim[0]*data_dsp_blk[0]*dh/1000., 0], vmin=0.08*vmin, vmax=0.08*vmax)
    cbar=plt.colorbar(seis,ax=ax)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(label='Pressure', size=10)
    for label in  ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(12)
    ax.set_title('Seismic Data',{'family': 'Times New Roman', 'weight':'normal', 'size':16})
    ax.set_xlabel('Position (km)',{'family': 'Times New Roman', 'weight':'normal', 'size':10})
    ax.set_ylabel('Time (s)',{'family': 'Times New Roman', 'weight':'normal', 'size':10})
    ax.vlines(x=[seis_rec1/100, seis_rec2/100, seis_rec3/100], ymin=0, ymax=2, colors=['r', 'b', 'k'], linestyles='--')
    fig.patch.set_facecolor("#C0C0C0") ##C5C9C7
    fig.patch.set_edgecolor("#C0C0C0") ##C5C9C7
    plt.tick_params(labelsize=10)
    plt.subplots_adjust(bottom=0.20,top=0.70,left=0.14,right=0.85)
    plt.savefig(SavePath+'Seismic', facecolor=fig.get_facecolor(), 
                        edgecolor=fig.get_edgecolor(), transparent=True)

#Display trace
def plot_trace(trace1, trace2, trace3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath):
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(7, 5), nrows=3, sharex=True, sharey=True, subplot_kw=dict(frameon=False))
    plt.suptitle("Trace of 3 receivers", fontsize=16, y=1.00)
    ax1.plot(trace1, '-', scaley=False, color="r", label='set1')
    ax1.get_yaxis().set_visible(True)
    ax1.get_xaxis().set_visible(False)
    ax2.plot(trace2, '-', color="b", label='set2')
    ax2.get_yaxis().set_visible(True)
    ax2.get_xaxis().set_visible(False)
    ax3.plot(trace3, '-', color="k", label='set3')
    ax3.get_yaxis().set_visible(True)
    for label in  ax1.get_xticklabels()+ax1.get_yticklabels():
        label.set_fontsize(12)
    for label in  ax2.get_xticklabels()+ax2.get_yticklabels():
        label.set_fontsize(12)
    for label in  ax3.get_xticklabels()+ax3.get_yticklabels():
        label.set_fontsize(12)
    locs, labels = plt.xticks()
    labels = [(item)/100 for item in locs]
    plt.xticks(locs[1:-1], labels[1:-1])
    line_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*line_labels)]
    fig.legend(lines, labels, fontsize=12)
    # fig.text(0.5, 0.001, "Time (s)", ha="center")
    # fig.text(0.001, 0.5, "Pressure", va="center", rotation="vertical")
    ax3.set_xlabel("Time (s)", {'family': 'Times New Roman', 'weight':'normal', 'size':14})
    ax2.set_ylabel("Pressure ", {'family': 'Times New Roman', 'weight':'normal', 'size':14})
    fig.set_facecolor("#C0C0C0")
    fig.set_edgecolor("#C0C0C0")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)
    plt.savefig(SavePath+'Trace', facecolor=fig.get_facecolor(), 
                        edgecolor=fig.get_edgecolor(), transparent=True)

#Display velocity profile
def plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath):
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(7, 5), nrows=3, sharex=True, sharey=True, subplot_kw=dict(frameon=False))
    plt.suptitle("Velocity Profile of 3 locations", fontsize=20, y=1.00)
    prof1 = np.flip(prof1)
    ax1.plot(prof1, '-', scaley=False, color="r", label='set1')
    ax1.get_yaxis().set_visible(True)
    ax1.get_xaxis().set_visible(False)
    prof2 = np.flip(prof2)
    ax2.plot(prof2, '-', color="b", label='set2')
    ax2.get_yaxis().set_visible(True)
    ax2.get_xaxis().set_visible(False)
    prof3 = np.flip(prof3)
    ax3.plot(prof3, '-', color="k", label='set3')
    ax3.get_yaxis().set_visible(True)
    for label in  ax1.get_xticklabels()+ax1.get_yticklabels():
        label.set_fontsize(12)
    for label in  ax2.get_xticklabels()+ax2.get_yticklabels():
        label.set_fontsize(12)
    for label in  ax3.get_xticklabels()+ax3.get_yticklabels():
        label.set_fontsize(12)
    locs, labels = plt.xticks()
    labels = [(item)/100 for item in locs]
    plt.xticks(locs[1:-1], labels[1:-1])
    line_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*line_labels)]
    fig.legend(lines, labels, loc=2, fontsize=12)
    ax3.set_xlabel("Depth (km)", {'family': 'Times New Roman', 'weight':'normal', 'size':16})
    ax2.set_ylabel("Velocity (m/s)", {'family': 'Times New Roman', 'weight':'normal', 'size':16})
    fig.set_facecolor("#C0C0C0")
    fig.set_edgecolor("#C0C0C0") 
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)
    plt.savefig(SavePath+'VelProfile', facecolor=fig.get_facecolor(), 
                            edgecolor=fig.get_edgecolor(), transparent=True)

def PlotOriginalPrediction(pd,label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath):
    PD = pd.reshape(label_dsp_dim[0],label_dsp_dim[1])
    fig2,ax=plt.subplots(figsize=(9, 5))
    # fig2.set_dpi(150)
    im2=ax.imshow(PD,extent=[0,label_dsp_dim[1]*label_dsp_blk[1]*dh/1000., \
                              0,label_dsp_dim[0]*label_dsp_blk[0]*dh/1000.],vmin=minvalue,vmax=maxvalue)
    divider = make_axes_locatable(ax)
    cax    = divider.append_axes("right",size="5%",pad=0.05)
    for label in  ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(10)   
    ax.set_xlabel('Position (km)',fontsize=10)
    ax.set_ylabel('Depth (km)',fontsize=10)
    ax.set_title('Prediction',{'family': 'Times New Roman','weight': 'normal','size': 16})
    ax.invert_yaxis()
    fig2.patch.set_facecolor("#C0C0C0") ##C5C9C7
    fig2.patch.set_edgecolor("#C0C0C0") ##C5C9C7
    cbar=plt.colorbar(im2,ax=ax,cax=cax)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(label='Velocity (m/s)', size=12)
    # plt.subplots_adjust(bottom=0.20,top=0.65,left=0.08,right=0.95)
    plt.savefig(SavePath+'PD', facecolor=fig2.get_facecolor(), 
                                edgecolor=fig2.get_edgecolor(), transparent=True)

#Display seismic data
def DisplayOriginalSeismic(seis, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath):
    vmin, vmax = np.min(seis), np.max(seis)
    fig, ax = plt.subplots(figsize=(4, 4))
    seis = ax.imshow(seis, cmap='gray', aspect="auto", extent=[0,data_dsp_dim[1]*data_dsp_blk[1]*dh/1000., \
                            data_dsp_dim[0]*data_dsp_blk[0]*dh/1000., 0], vmin=0.08*vmin, vmax=0.08*vmax)
    cbar=plt.colorbar(seis,ax=ax)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(label='Pressure', size=10)
    for label in ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(12)
    ax.set_title('Seismic Data',{'family': 'Times New Roman', 'weight':'normal', 'size':16})
    ax.set_xlabel('Position (km)',{'family': 'Times New Roman', 'weight':'normal', 'size':10})
    ax.set_ylabel('Time (s)',{'family': 'Times New Roman', 'weight':'normal', 'size':10})
    fig.patch.set_facecolor("#C0C0C0") ##C5C9C7
    fig.patch.set_edgecolor("#C0C0C0") ##C5C9C7
    plt.tick_params(labelsize=10)
    plt.subplots_adjust(bottom=0.20,top=0.70,left=0.14,right=0.85)
    plt.savefig(SavePath+'Seismic', facecolor=fig.get_facecolor(), 
                        edgecolor=fig.get_edgecolor(), transparent=True)

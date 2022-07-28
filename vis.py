import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


# Load colormap for velocity map visualization
rainbow_cmap = ListedColormap(np.load('/home/pi/Desktop/exp/fcnvmb/UPFWI/src/rainbow256.npy'))

def plot_kimb_velocity(output, target, path, vmin=None, vmax=None):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    if vmin is None or vmax is None:
        vmax, vmin = np.max(target), np.min(target)

    im = ax.imshow(output, cmap=rainbow_cmap, extent=[0, 401, 141, 0], vmin=vmin, vmax=vmax)
    
    #for axis in ax:
        #axis.set_xticks(range(0, 200, 10)) #(0, 40, 10)
        #axis.set_xticklabels(range(0, 400, 100)) #(0, 400, 100)
        #axis.set_yticks(range(0, 100, 10)) #(0, 60, 10)
        #axis.set_yticklabels(range(0, 140, 20)) #(0, 600, 100)
        # axis.set_xticks(range(0, 60, 10))
        # axis.set_xticklabels(range(0, 600, 100))
        # axis.set_yticks(range(0, 40, 10))
        # axis.set_yticklabels(range(0, 400, 100))
    plt.rcParams.update({'font.size': 14})
    ax.set_title('Offset (m)', y=1.07, fontsize=13)
    ax.set_ylabel('Depth (m)', fontsize=13)

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, label='Velocity(m/s)')
    cbar.ax.tick_params(labelsize=12)
    plt.savefig(path)
    plt.close('all')

def plot_single_velocity(label, path):
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    vmax, vmin = np.max(label), np.min(label)
    im = ax.matshow(label, cmap=rainbow_cmap, vmin=vmin, vmax=vmax)

    nx = label.shape[0]
    ax.set_aspect(aspect=1)
    ax.set_xticks(range(0, nx, int(150//(1050/nx)))[:7])
    ax.set_xticklabels(range(0, 1050, 150))
    ax.set_yticks(range(0, nx, int(150//(1050/nx)))[:7])
    ax.set_yticklabels(range(0, 1050, 150))
    ax.set_title('Offset (m)', y=1.08)
    ax.set_ylabel('Depth (m)', fontsize=18)

    fig.colorbar(im, ax=ax, shrink=1.0, label='Velocity(m/s)', labelsize=13)
    plt.savefig(path)
    plt.close('all')

# def plot_seismic(output, target, path, vmin=-1e-5, vmax=1e-5):
#     fig, ax = plt.subplots(1, 3, figsize=(15, 6))
#     im = ax[0].matshow(output, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax)
#     ax[0].set_title('Prediction')
#     ax[1].matshow(target, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax)
#     ax[1].set_title('Ground Truth')
#     ax[2].matshow(output - target, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax)
#     ax[2].set_title('Difference')
#     fig.colorbar(im, ax=ax, format='%.1e')
#     plt.savefig(path)
#     plt.close('all')

def plot_seismic(output, target, path, vmin=-1e-5, vmax=1e-5):
    # fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    aspect = output.shape[1]/output.shape[0]
    im = ax[0].matshow(target, aspect=aspect, cmap='gray', vmin=vmin, vmax=vmax)
    # ax[0].set_title('Prediction')
    ax[1].matshow(output, aspect=aspect, cmap='gray', vmin=vmin, vmax=vmax)
    # ax[1].set_title('Ground Truth')
    # ax[2].matshow(output - target, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax)
    # ax[2].set_title('Difference')
    
    for axis in ax:
        axis.set_xticks(range(0, 70, 10))
        axis.set_xticklabels(range(0, 1050, 150))
        axis.set_title('Offset (m)', y=1.1)
        axis.set_ylabel('Time (ms)', fontsize=12)
    
    # fig.colorbar(im, ax=ax, shrink=1.0, pad=0.01, label='Amplitude')
    fig.colorbar(im, ax=ax, shrink=0.75, label='Amplitude')
    plt.savefig(path)
    plt.close('all')

def plot_kimb_seismic(data, path):
    # plt.rcParams.update({'font.size': 18})
    vmin, vmax = np.min(data), np.max(data)
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    im = ax.imshow(data, aspect='auto', cmap='gray', extent=[0, data.shape[1] * 40/1000., data.shape[0] * 0.002, 0], vmin=vmin*0.4, vmax=vmax*0.4)
    plt.tick_params(labelsize=6)
    for label in  ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(14)
    ax.set_title('Seismic Data',{'family': 'Times New Roman','weight': 'normal','size': 24,})
    ax.set_xlabel('Offset (km)', fontsize=18)
    ax.set_ylabel('Time (s)', fontsize=18)
    fig.patch.set_facecolor("#C0C0C0") 
    fig.patch.set_edgecolor("#C0C0C0")
    cbar=plt.colorbar(im, ax=ax, shrink=1.0, pad=0.05)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(label='Amplitude', size=14)
    plt.savefig(path, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())
    plt.close('all')

# Display seismic data
def display_annot_seismic(data, seis_rec1, seis_rec2, seis_rec3, path):
    # plt.rcParams.update({'font.size': 18})
    vmin, vmax = np.min(data), np.max(data)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(data, cmap='gray', aspect="auto", extent=[0, data.shape[1] * 40/1000., data.shape[0] * 0.002, 0], vmin=0.4*vmin, vmax=0.4*vmax)
    plt.tick_params(labelsize=6)
    for label in  ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(14)
    ax.set_title('Seismic Data',{'family': 'Times New Roman','weight': 'normal','size': 24,})
    ax.set_xlabel('Offset (km)', fontsize=18)
    ax.set_ylabel('Time (s)', fontsize=18)
    ax.vlines(x=[seis_rec1/1000, seis_rec2/1000, seis_rec3/1000], ymin=0, ymax=2.5, colors=['r', 'b', 'k'], linestyles='solid', lw=2)
    fig.patch.set_facecolor("#C0C0C0") 
    fig.patch.set_edgecolor("#C0C0C0")
    cbar=plt.colorbar(im, ax=ax, shrink=1.0, pad=0.05)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(label='Amplitude', size=14)
    plt.savefig(path, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())
    plt.close('all')

def plot_kimb_trace(trace1, trace2, trace3, path):
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(7, 5), nrows=3, sharex=True, sharey=True, subplot_kw=dict(frameon=False))
    plt.suptitle("Trace of 3 receivers", fontsize=16, y=1.00)
    ax1.plot(trace1, scaley=False, color="r", label='set1')
    ax1.get_yaxis().set_visible(True)
    ax1.get_xaxis().set_visible(False)
    ax2.plot(trace2, color="b", label='set2')
    ax2.get_yaxis().set_visible(True)
    ax2.get_xaxis().set_visible(False)
    ax3.plot(trace3, color="k", label='set3')
    ax3.get_yaxis().set_visible(True)
    for label in  ax1.get_xticklabels()+ax1.get_yticklabels():
        label.set_fontsize(12)
    for label in  ax2.get_xticklabels()+ax2.get_yticklabels():
        label.set_fontsize(12)
    for label in  ax3.get_xticklabels()+ax3.get_yticklabels():
        label.set_fontsize(12)
    locs, labels = plt.xticks()
    labels = [(item)*0.002 for item in locs]
    plt.xticks(locs[1:-1], labels[1:-1])
    line_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*line_labels)]
    fig.legend(lines, labels, fontsize=12)
    ax3.set_xlabel("Time (s)", {'family': 'Times New Roman', 'weight':'normal', 'size':13})
    ax2.set_ylabel("Pressure ", {'family': 'Times New Roman', 'weight':'normal', 'size':13})
    fig.set_facecolor("#C0C0C0")
    fig.set_edgecolor("#C0C0C0")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)
    plt.savefig(path, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor(), transparent=True)

def plot_kimb_profile(prof1, prof2, prof3, path):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, sharey=True, subplot_kw=dict(frameon=False))
    plt.suptitle("Velocity Profile of 3 receivers", fontsize=14, y=1.00)
    ax1.plot(prof1, scaley=False, color="r", label='set1')
    ax1.get_yaxis().set_visible(True)
    ax1.get_xaxis().set_visible(False)
    ax2.plot(prof2, color="b", label='set2')
    ax2.get_yaxis().set_visible(True)
    ax2.get_xaxis().set_visible(False)
    ax3.plot(prof3, color="k", label='set3')
    ax3.get_yaxis().set_visible(True)
    for label in  ax1.get_xticklabels()+ax1.get_yticklabels():
        label.set_fontsize(12)
    for label in  ax2.get_xticklabels()+ax2.get_yticklabels():
        label.set_fontsize(12)
    for label in  ax3.get_xticklabels()+ax3.get_yticklabels():
        label.set_fontsize(12)
    locs, labels = plt.xticks()
    line_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*line_labels)]
    fig.legend(lines, labels, loc=2, fontsize=12)
    fig.text(0.5, 0.001, "Depth (m)", fontsize=14, ha="center")
    fig.text(0.03, 0.5, "Velocity (m/s)", fontsize=14, va="center", rotation="vertical")
    fig.set_facecolor("#C0C0C0")
    fig.set_edgecolor("#C0C0C0") 
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)
    plt.savefig(path, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor(), transparent=True)
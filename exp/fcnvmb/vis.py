import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Load colormap for velocity map visualization
rainbow_cmap = ListedColormap(np.load('/home/pi/Desktop/exp/fcnvmb/UPFWI/src/rainbow256.npy'))

def plot_kimb_velocity(output, target, path, vmin=None, vmax=None):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    if vmin is None or vmax is None:
        vmin, vmax = np.min(target), np.max(target)
    im = ax.imshow(output, cmap=rainbow_cmap, extent=[0, target.shape[1] * 1/100., target.shape[0] * 2.5/141., 0], vmin=1500, vmax=4500)
    # im = ax.imshow(output, cmap=rainbow_cmap, extent=[0, 4, 2.5, 0], vmin=1500, vmax=4500)
    divider = make_axes_locatable(ax)
    cax    = divider.append_axes("right",size="5%",pad=0.05)
    for label in  ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(10)
    ax.set_title('Prediction',{'family': 'Times New Roman','weight': 'normal','size': 16})
    ax.set_xlabel('Position (km)', fontsize=10)
    ax.set_ylabel('Depth (km)', fontsize=10)
    fig.patch.set_facecolor("#C0C0C0") 
    fig.patch.set_edgecolor("#C0C0C0")
    cbar = fig.colorbar(im, ax=ax, cax=cax)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(label='Velocity (m/s)', size=12)
    plt.savefig(path, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())
    plt.close('all')

def plot_annot_kimb_velocity(output, target, path, vel_rec1, vel_rec2, vel_rec3, vmin=None, vmax=None):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    if vmin is None or vmax is None:
        vmin, vmax = np.min(target), np.max(vmax)
    im = ax.imshow(output, cmap=rainbow_cmap, extent=[0, target.shape[1] * 1/100., target.shape[0] * 2.5/141., 0], vmin=1500, vmax=4500)
    divider = make_axes_locatable(ax)
    cax    = divider.append_axes("right",size="5%",pad=0.05)
    for label in  ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(10)
    ax.set_title('Prediction',{'family': 'Times New Roman','weight': 'normal','size': 16})
    ax.set_xlabel('Position (km)', fontsize=10)
    ax.set_ylabel('Depth (km)', fontsize=10)
    ax.vlines(x=[vel_rec1/1000, vel_rec2/1000, vel_rec3/1000], ymin=0, ymax=2.5, colors=['r', 'b', 'k'], linestyles='solid', lw=2)
    fig.patch.set_facecolor("#C0C0C0") 
    fig.patch.set_edgecolor("#C0C0C0")
    cbar = fig.colorbar(im, ax=ax, cax=cax)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(label='Velocity (m/s)', size=12)
    plt.savefig(path, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())
    plt.close('all')

def plot_kimb_seismic(data, path):
    vmin, vmax = np.min(data), np.max(data)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    im = ax.imshow(data, aspect='auto', cmap='gray', extent=[0, data.shape[1] * 40/1000., data.shape[0] * 0.002, 0], vmin=vmin*0.4, vmax=vmax*0.4)
    for label in  ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(9)
    ax.set_title('Seismic Data',{'family': 'Times New Roman','weight': 'normal','size': 16})
    ax.set_xlabel('Offset (km)', fontsize=10)
    ax.set_ylabel('Time (s)', fontsize=10)
    fig.patch.set_facecolor("#C0C0C0") 
    fig.patch.set_edgecolor("#C0C0C0")
    cbar=plt.colorbar(im, ax=ax, shrink=1.0, pad=0.05)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(label='Amplitude', size=10)
    plt.savefig(path, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())
    plt.close('all')

# Display seismic data
def display_annot_seismic(data, seis_rec1, seis_rec2, seis_rec3, path):
    vmin, vmax = np.min(data), np.max(data)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(data, cmap='gray', aspect="auto", extent=[0, data.shape[1] * 40/1000., data.shape[0] * 0.002, 0], vmin=0.4*vmin, vmax=0.4*vmax)
    for label in  ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(9)
    ax.set_title('Seismic Data',{'family': 'Times New Roman','weight': 'normal','size': 16})
    ax.set_xlabel('Offset (km)', fontsize=10)
    ax.set_ylabel('Time (s)', fontsize=10)
    ax.vlines(x=[seis_rec1/1000, seis_rec2/1000, seis_rec3/1000], ymin=0, ymax=2.5, colors=['r', 'b', 'k'], linestyles='solid', lw=2)
    fig.patch.set_facecolor("#C0C0C0") 
    fig.patch.set_edgecolor("#C0C0C0")
    cbar=plt.colorbar(im, ax=ax, shrink=1.0, pad=0.05)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(label='Amplitude', size=10)
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
    ax2.set_ylabel("Amplitude", {'family': 'Times New Roman', 'weight':'normal', 'size':13})
    fig.set_facecolor("#C0C0C0")
    fig.set_edgecolor("#C0C0C0")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)
    plt.savefig(path, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor(), transparent=True)

def plot_kimb_profile(prof1, prof2, prof3, path):
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(5,4), nrows=3, sharex=True, sharey=True, subplot_kw=dict(frameon=False))
    plt.suptitle("Velocity Profile of 3 locations", fontsize=14, y=1.00)
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
    labels = [(item) * 2.5/141 for item in locs]
    labels = ["{:.1f}".format(value) for value in labels]
    plt.xticks(locs[1:-1], labels[1:-1])
    line_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*line_labels)]
    fig.legend(lines, labels, loc=2, fontsize=12)
    ax3.set_xlabel("Depth (km)", {'family': 'Times New Roman', 'weight':'normal', 'size':12})
    ax2.set_ylabel("Velocity (m/s)", {'family': 'Times New Roman', 'weight':'normal', 'size':12})
    fig.set_facecolor("#C0C0C0")
    fig.set_edgecolor("#C0C0C0") 
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)
    plt.savefig(path, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor(), transparent=True)
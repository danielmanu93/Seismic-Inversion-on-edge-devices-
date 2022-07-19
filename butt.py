# -*- coding: utf-8 -*-
from fileinput import filename
from tkinter import filedialog, ttk
from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
import os
import customtkinter
import sys
import numpy as np
import FCNVMB_test
from ParamConfig import *
from PathConfig import *
from LibConfig import *
import datetime
import time
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch import nn
import torchvision
from torchvision import transforms
import numpy as np
import utils
import network
from vis import plot_kimb_velocity, plot_kimb_seismic, plot_kimb_trace, plot_kimb_profile
from dataset import FWIDataset
import transforms as T
import pytorch_ssim
import math
import tkinter.scrolledtext as tkscrolled
import tkinter.font as font

root = Tk()
# root = tk.Toplevel()
root.attributes('-zoomed', True)
# w, h = root.winfo_screenwidth(), root.winfo_screenheight()
# root.geometry('%dx%d+0+0' % (w, h))
root.title("Inversion GUI App")
canvas = tk.Canvas(root, height=1900, width=1900)
canvas.pack()

def restart():
    canvas1.delete("all")
    canvas2.delete("all")
    canvas3.delete(canv_img3)
    canvas4.delete(canv_img4)
    screen.delete('1.0', END)
  
menubar = Menu(root, font=("Times", "10", "bold"))
filemenu = Menu(menubar, tearoff=0)
newImg = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/exp/fcnvmb/images/new.png').resize((20, 20), Image.ANTIALIAS))
filemenu.add_command(label=" New", image=newImg, compound=LEFT, command=())
openImg = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/exp/fcnvmb/images/open.png').resize((20, 20), Image.ANTIALIAS))
filemenu.add_command(label=" Open...", image=openImg, compound=LEFT, command=())
saveImg = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/exp/fcnvmb/images/save.png').resize((20, 20), Image.ANTIALIAS))
filemenu.add_command(label=" Save", image=saveImg, compound=LEFT,  command=())
filemenu.add_separator()
resImg = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/exp/fcnvmb/images/restart.png').resize((20, 20), Image.ANTIALIAS))
filemenu.add_command(label=" Restart", image=resImg, compound=LEFT, command=restart)
exitImg = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/exp/fcnvmb/images/close.png').resize((20, 20), Image.ANTIALIAS))
filemenu.add_command(label=" Exit", image=exitImg, compound=LEFT, command=root.destroy)
menubar.add_cascade(label="File", menu=filemenu)

data = None
def open_salt():
    global data
    filename = filedialog.askopenfilename(initialdir="/home/pi/Desktop/exp/fcnvmb/data/test_data/SaltData/georec_test", title="Select A Salt Data", 
                               filetypes=(("numpy files", "*.npy"), ("all files", "*.*")))
    # screen.insert(INSERT, " ", "\n")
    screen.insert(INSERT, "******************** Selected Salt Data Loaded ********************" + '\n') 
    # data = np.load(filename)

    for file in filename:
        screen.insert(END, file, '\n')
        # data = np.load(file)
            
        # screen.insert(INSERT, data, "\n")
        # print(data, end="")
# open_salt()
# print(data)

def open_kimb():
    dialog = filedialog.askopenfilename(initialdir="/home/pi/Desktop/exp/fcnvmb/data/kimb_test_data", title="Select A Kimberlina Data", 
                               filetypes=(("numpy files", "*.npy"), ("all files", "*.*")))
    screen.insert(INSERT, "******************** Selected Kimberlina Data Loaded ********************" + '\n') 
    
    for data in dialog:
        screen.insert(INSERT, data, "\n")

datamenu = Menu(menubar, tearoff=0)
saltImg = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/exp/fcnvmb/images/salt.png').resize((80, 20), Image.ANTIALIAS))
datamenu.add_command(image=saltImg, command=open_salt)
kimbImg = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/exp/fcnvmb/images/kim.png').resize((80, 20), Image.ANTIALIAS))
datamenu.add_command(image=kimbImg, command=open_kimb)
menubar.add_cascade(label="Select Data", menu=datamenu)

frame1 = customtkinter.CTkFrame(master=root, width=480, height=450, fg_color = "gray80", corner_radius=15, border_width=1, border_color="gray")
frame1.place(x=2, y=287)
frame1.pack_propagate(0)
seis = customtkinter.CTkLabel(master=frame1, text="Seismic Data", text_font=("helvetica", 13), fg_color="gray70", corner_radius=8)
seis.pack(pady=5)
canvas1 = tk.Canvas(frame1, bg="gray80", bd=1, height=400, width=470, highlightbackground="gray80")
canvas1.pack()

frame2 = customtkinter.CTkFrame(master=root, width=480, height=450, fg_color = "gray80", corner_radius=15, border_width=1, border_color="gray")
frame2.place(x=484, y=287)
frame2.pack_propagate(0)
seis_trace = customtkinter.CTkLabel(master=frame2, text="Seismic Trace", text_font=("helvetica", 13), fg_color="gray70", corner_radius=8)
seis_trace.pack(pady=5)
canvas3 = tk.Canvas(frame2, bg="gray80", bd=1, height=400, width=470, highlightbackground="gray80")
canvas3.pack()

frame3 = customtkinter.CTkFrame(master=root, width=480, height=450, fg_color = "gray80", corner_radius=15, border_width=1, border_color="gray")
frame3.place(x=966, y=287)
frame3.pack_propagate(0)
pred_model = customtkinter.CTkLabel(master=frame3, text="Predicted Velocity Model", text_font=("helvetica", 13), fg_color="gray70", corner_radius=8)
pred_model.pack(pady=5)
canvas2 = tk.Canvas(frame3, bg="gray80", bd=1, height=400, width=470, highlightbackground="gray80")
canvas2.pack()

frame4 = customtkinter.CTkFrame(master=root, width=470, height=450, fg_color = "gray80", corner_radius=15, border_width=1, border_color="gray")
frame4.place(x=1448, y=287)
frame4.pack_propagate(0)
vel_profile = customtkinter.CTkLabel(master=frame4, text="Velocity Profile", text_font=("helvetica", 13), fg_color="gray70", corner_radius=8)
vel_profile.pack(pady=5)
canvas4 = tk.Canvas(frame4, bg="gray80", bd=1, height=400, width=470, highlightbackground="gray80")
canvas4.pack()

frame5 = customtkinter.CTkFrame(master=root,  width=962, height=257, fg_color = "gray80", corner_radius=15, border_width=1, border_color="gray")
frame5.place(x=483, y=739)
frame5.pack_propagate(0)
out_screen = customtkinter.CTkLabel(master=frame5, text="Output Display", text_font=("helvetica", 13), fg_color="gray70", corner_radius=8)
out_screen.pack(pady=3)

seis_labelFrame = LabelFrame(root, text=" Seismic ", font=("helvetica", 13), labelanchor="n", height=280, width=480, bg="gray80", bd=4, relief=GROOVE)
seis_labelFrame.place(x=3, y=7)

pred_labelFrame = LabelFrame(root, text=" Velocity Model ", font=("helvetica", 13), labelanchor="n", height=280, width=482, bg="gray80", bd=4, relief=GROOVE)
pred_labelFrame.place(x=965, y=7)

trace_labelFrame = LabelFrame(root, text=" Trace ", font=("helvetica", 13), labelanchor="n", height=280, width=483, bg="gray80", bd=4, relief=GROOVE)
trace_labelFrame.place(x=483, y=7)

vel_labelFrame = LabelFrame(root, text=" Velocity Profile ", font=("helvetica", 13), labelanchor="n", height=280, width=472, bg="gray80", bd=4, relief=GROOVE)
vel_labelFrame.place(x=1448, y=7)

data_labelFrame = LabelFrame(root, text=" Seismic and Velocity with Receiver Annotations ", font=("helvetica", 13), labelanchor="n", height=258, width=480, bg="gray80", bd=4, relief=GROOVE)
data_labelFrame.place(x=2, y=739)

unm = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/exp/fcnvmb/images/unm.PNG').resize((80, 70), Image.ANTIALIAS))
lanl = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/exp/fcnvmb/images/lanl.PNG').resize((80, 70), Image.ANTIALIAS))
gmu = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/exp/fcnvmb/images/gmu.PNG').resize((80, 70), Image.ANTIALIAS))
infoFrame = customtkinter.CTkFrame(master=root,  width=470, height=257, fg_color = "gray80", corner_radius=15, border_width=1, border_color="gray")
infoFrame.place(x=1448, y=739)
infoscreen = Text(root, bd=2, wrap="word")
infoscreen.tag_configure("center", justify='center')
infoscreen.place(x=1452, y=747, width=459, height=240)
infoscreen.configure(font=("helvetica", 10, "bold"))
infoscreen.insert(INSERT, "Designed by:" + "\n")
infoscreen.insert(INSERT, " " + "\n")
infoscreen.insert(INSERT, "Daniel Manu, Petro Mushidi Tshakwanda, Lin Youzuo, Weiwen Jiang, Lei Yang" + '\n')
infoscreen.insert(INSERT, " " + "\n")
infoscreen.insert(INSERT, "University of New Mexico, Los Alamos National Laboratory, George Mason University" + '\n')
infoscreen.insert(INSERT, " " + "\n")
infoscreen.tag_add("center", "1.0", "end")
infoscreen.image_create(tk.END, image = unm)
infoscreen.image_create(tk.END, image = lanl)
infoscreen.image_create(tk.END, image = gmu)
infoscreen.insert(INSERT, " " + "\n")
infoscreen.insert(INSERT, " " + "\n")
infoscreen.insert(INSERT,"\u00A9 2022 Copyright \u00AE\u2122" + "\n")
infoscreen.config(state=DISABLED)

def PredImage():
    orig_screen_disp()
    global pd, PD, pred_button, canv_img2
    PD = Image.open('/home/pi/Desktop/exp/fcnvmb/results/SaltResults/PD.png')
    pd = ImageTk.PhotoImage(PD.resize((400, 390), Image.ANTIALIAS))
    canv_img2 = canvas2.create_image(235, 207, image=pd, anchor=CENTER)

def AnnotPredImage():
    screen_disp()
    global annot_pd, annot_PD, annot_pred_button, canv_img2
    annot_PD = Image.open('/home/pi/Desktop/exp/fcnvmb/results/SaltResults/PD.png')
    annot_pd = ImageTk.PhotoImage(annot_PD.resize((400, 390), Image.ANTIALIAS))
    canv_img2 = canvas2.create_image(235, 207, image=annot_pd, anchor=CENTER)

def AnnotNoisePredImage():
    noise_screen_disp()
    global annot_pd_noise, annot_PD_noise, annot_pred_noise_button, canv_img2
    annot_PD_noise = Image.open('/home/pi/Desktop/exp/fcnvmb/results/SaltResults/PD.png')
    annot_pd_noise = ImageTk.PhotoImage(annot_PD_noise.resize((400, 390), Image.ANTIALIAS))
    canv_img2 = canvas2.create_image(235, 207, image=annot_pd_noise, anchor=CENTER)

def NoisePredImage():
    orig_noise_screen_disp()
    global pd_noise, PD_noise, noiseUNet_button, canv_img2
    PD_noise = Image.open('/home/pi/Desktop/exp/fcnvmb/results/SaltResults/PD.png')
    pd_noise = ImageTk.PhotoImage(PD_noise.resize((400, 390), Image.ANTIALIAS))
    canv_img2 = canvas2.create_image(235, 207, image=pd_noise, anchor=CENTER)

def KimbPredImage():
    orig_kimb_screen_disp()
    global kimb_pd, kimb_PD, model_button, canv_img2
    kimb_PD = Image.open('/home/pi/Desktop/exp/fcnvmb/results/SaltResults/PD.png')
    kimb_pd = ImageTk.PhotoImage(PD.resize((400, 390), Image.ANTIALIAS))
    canv_img2 = canvas2.create_image(235, 207, image=kimb_pd, anchor=CENTER)

def SeisImage():
    global seis, Seis, receive_button, canv_img1
    Seis = Image.open('/home/pi/Desktop/exp/fcnvmb/results/SaltResults/Seismic.png')
    seis = ImageTk.PhotoImage(Seis.resize((400, 390), Image.ANTIALIAS))
    canv_img1 = canvas1.create_image(235, 207, image=seis, anchor=CENTER)

def KimbSeisImage():
    global kimb_seis, kimb_Seis, receive_kimb_button, canv_img1
    kimb_Seis = Image.open('/home/pi/Desktop/exp/fcnvmb/results/SaltResults/Seismic.png')
    kimb_seis = ImageTk.PhotoImage(kimb_Seis.resize((400, 390), Image.ANTIALIAS))
    canv_img1 = canvas1.create_image(235, 207, image=kimb_seis, anchor=CENTER)

def AnnotSeisImage():
    global annot_seis, annot_Seis, annot_seis_button, canv_img1
    annot_Seis = Image.open('/home/pi/Desktop/exp/fcnvmb/results/SaltResults/Seismic.png')
    annot_seis = ImageTk.PhotoImage(annot_Seis.resize((400, 390), Image.ANTIALIAS))
    canv_img1 = canvas1.create_image(235, 207, image=annot_seis, anchor=CENTER)

def TraceImage():
    global trac, trace_label, Trace, trace_button, canv_img3
    Trace = Image.open('/home/pi/Desktop/exp/fcnvmb/results/SaltResults/Trace.png')
    trac = ImageTk.PhotoImage(Trace.resize((420, 390), Image.ANTIALIAS))
    canv_img3 = canvas3.create_image(235, 207, image=trac, anchor=CENTER)

def ProfileImage():
    global vel, vel_label, Vel, vel_button, canv_img4
    Vel = Image.open('/home/pi/Desktop/exp/fcnvmb/results/SaltResults/VelProfile.png')
    vel = ImageTk.PhotoImage(Vel.resize((420, 390), Image.ANTIALIAS))
    canv_img4 = canvas4.create_image(235, 207, image=vel, anchor=CENTER)

def zoom_seis(zoom):
    global seisImg
    newsize = (Seis.size[0]* int(zoom), 
                Seis.size[1]*int(zoom))
    scaledseis = Seis.resize(newsize, Image.LINEAR)
    seisImg = ImageTk.PhotoImage(scaledseis)
    canvas1.itemconfig(canv_img1, image=seisImg)

def zoom_annot_seis(zoom):
    global annotseisImg
    newsize = (annot_Seis.size[0]* int(zoom), 
            annot_Seis.size[1]*int(zoom))
    scaledAnnotseis = annot_Seis.resize(newsize, Image.LINEAR)
    annotseisImg = ImageTk.PhotoImage(scaledAnnotseis)
    canvas1.itemconfig(canv_img1, image=annotseisImg)

def zoom_pd(zoom):
    global pdImg
    newsize = (PD.size[0]* int(zoom), 
                PD.size[1]*int(zoom))
    scaledPD = PD.resize(newsize, Image.LINEAR)
    pdImg = ImageTk.PhotoImage(scaledPD)
    canvas2.itemconfig(canv_img2, image=pdImg)

def zoom_annot_pd(zoom):
    global annotpdImg
    newsize = (annot_PD.size[0]* int(zoom), 
            annot_PD.size[1]*int(zoom))
    scaledAnnotPD = annot_PD.resize(newsize, Image.LINEAR)
    annotpdImg = ImageTk.PhotoImage(scaledAnnotPD)
    canvas2.itemconfig(canv_img2, image=annotpdImg)
    # return annotpdImg

def zoom_annot_pd_noise(zoom):
    global annotpdnoiseImg
    newsize = (annot_PD_noise.size[0]* int(zoom), 
            annot_PD_noise.size[1]*int(zoom))
    scaledAnnotPDNoise = annot_PD_noise.resize(newsize, Image.LINEAR)
    annotpdnoiseImg = ImageTk.PhotoImage(scaledAnnotPDNoise)
    canvas2.itemconfig(canv_img2, image=annotpdnoiseImg)

def zoom_kimb_pd(zoom):
    global kimbpdImg
    newsize = (kimb_PD.size[0]* int(zoom), 
                kimb_PD.size[1]*int(zoom))
    scaledKimbPD = kimb_PD.resize(newsize, Image.LINEAR)
    kimbpdImg = ImageTk.PhotoImage(scaledKimbPD)
    canvas2.itemconfig(canv_img2, image=kimbpdImg)
 
def move_from_1(event):
    canvas1.scan_mark(event.x, event.y)

def move_to_1(event):
    canvas1.scan_dragto(event.x, event.y, gain=1)

def move_from_2(event):
    canvas2.scan_mark(event.x, event.y)

def move_to_2(event):
    canvas2.scan_dragto(event.x, event.y, gain=1)

canvas1.bind('<ButtonPress-1>', move_from_1)
canvas1.bind('<B1-Motion>', move_to_1)

canvas2.bind('<ButtonPress-3>', move_from_2)
canvas2.bind('<B3-Motion>', move_to_2)

def TestingPD():
    PredImage()

def ReceiveCall():
    orig_seis()
    SeisImage()

def TraceCall():
    TraceImage()

def VelocityCall():
    ProfileImage()

def NoisePD():
    NoisePredImage()

def AnnotTestingPD():
    AnnotPredImage()

def AnnotTestingPDNoise():
    AnnotNoisePredImage()

def AnnotReceiverCall():
    seis_with_annot()
    AnnotSeisImage()

def zoom_seismic(zoom=0):
    zoom_seis(zoom)
    zoom_annot_seis(zoom)
    
def zoom_pred(zoom=0):
    zoom_pd(zoom)
    zoom_annot_pd(zoom)
    zoom_annot_pd_noise(zoom)
    
img1 = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/exp/fcnvmb/images/zoom1.png').resize((25, 25), Image.ANTIALIAS))
img2 = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/exp/fcnvmb/images/zoom2.png').resize((25, 25), Image.ANTIALIAS))
img3 = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/exp/fcnvmb/images/receiver.png').resize((35, 30), Image.ANTIALIAS))
img4 = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/exp/fcnvmb/images/annot_no_noise.png').resize((30, 25), Image.ANTIALIAS))
img5 = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/exp/fcnvmb/images/data-receive.png').resize((40, 30), Image.ANTIALIAS))
img6 = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/exp/fcnvmb/images/prediction.png').resize((40, 30), Image.ANTIALIAS))
img7 = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/exp/fcnvmb/images/trace.JPG').resize((40, 30), Image.ANTIALIAS))
img8 = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/exp/fcnvmb/images/vel_prof.JPG').resize((40, 30), Image.ANTIALIAS))
img9 = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/exp/fcnvmb/images/noise.png').resize((100, 30), Image.ANTIALIAS))
img10 = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/exp/fcnvmb/images/annot_seis.png').resize((30, 25), Image.ANTIALIAS))
img11 = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/exp/fcnvmb/images/annot_noise_aware.png').resize((30, 25), Image.ANTIALIAS))
noise_aware_img =  ImageTk.PhotoImage(Image.open('/home/pi/Desktop/exp/fcnvmb/images/noise_aware.PNG').resize((80, 25), Image.ANTIALIAS))
inv = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/exp/fcnvmb/images/inv.png').resize((80, 25), Image.ANTIALIAS))

seis_scaler_label = customtkinter.CTkLabel(master=root, text="Zoom Seismic ", text_font=("helvetica", 13), width=220, image=img1, compound=RIGHT,  
                                        fg_color="gray80", corner_radius=1).place(x=132, y=105)

pd_scaler_label = customtkinter.CTkLabel(master=root, text="Zoom Prediction ", text_font=("helvetica", 13), image=img2, compound=RIGHT,  
                                        fg_color="gray80", corner_radius=1).place(x=1125, y=105)

var1 = StringVar()
seis_scale = tk.Scale(root, variable=var1, font=("helvetica", 11), orient='horizontal', bg = "gray80", bd=1,
                        from_=1, to=5, length=200, resolution=1, command=zoom_seismic)
seis_scale.place(x=138, y=135)

var2 = StringVar()
pd_scale = tk.Scale(root, variable=var2, font=("helvetica", 11), orient='horizontal', bg = "gray80", bd=1, 
                        from_=1, to=5, length=200, resolution=1, command=zoom_pred)
pd_scale.place(x=1103, y=135) 

noise_scaler_label = customtkinter.CTkLabel(master=root, text="   Noise (dB)   ", text_font=("helvetica", 13), width=100, image=img9, compound=RIGHT, 
                                    fg_color="gray80", corner_radius=1).place(x=602, y=105)

net = FCNVMB_test.net
test_set,label_set,data_dsp_dim,label_dsp_dim = DataLoad_Test(test_size=TestSize,test_data_dir=test_data_dir, \
                                                            data_dim=DataDim,in_channels=Inchannels, \
                                                            model_dim=ModelDim,data_dsp_blk=data_dsp_blk, \
                                                            label_dsp_blk=label_dsp_blk,start=121, \
                                                            datafilename=datafilename,dataname=dataname, \
                                                            truthfilename=truthfilename,truthname=truthname)
test        = data_utils.TensorDataset(torch.from_numpy(test_set),torch.from_numpy(label_set))
test_loader = data_utils.DataLoader(test,batch_size=1,shuffle=False)

def orig_seis():

    global noiseImg1, noiseImg2, noiseImg3, noiseImg4, noiseImg5, noiseImg6, noiseImg7

    font2 = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 17,
        }
    font3 = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 21,
        }

    images, labels = iter(test_loader).next()
    images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
    labels = labels.view(TestBatchSize,Nclasses,label_dsp_dim[0],label_dsp_dim[1])

    if (check_1.get()) == 0:

        if (scale.get()) == 0:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            # set the target SNR
            target_SNR_dB = 0

            # compute power of image values
            image_power = images ** 2

            # compute average image and convert to dB
            avg_image_power = np.mean(image_power)
            avg_image_dB = 10 * np.log10(avg_image_power)

            # compute noise and convert to  dB
            avg_noiseImg_dB = avg_image_dB - target_SNR_dB
            avg_noiseImg_power = 10 ** (avg_noiseImg_dB / 10)

            # generate sample of white noise
            mean_noise = 0
            noiseImg1 = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))

            images = images + noiseImg1
            images = torch.from_numpy(images)
            images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
            images = images.to(torch.float32)

            seis = images.view(Inchannels, data_dsp_dim[0], data_dsp_dim[1])
            seis1 = seis.numpy()
            seis   = seis[1,:,:].numpy()

            trace1 = seis1[1,:,50]
            trace2 = seis1[1,:,150]
            trace3 = seis1[1,:,250]

            DisplayOriginalSeismic(seis, data_dsp_dim, data_dsp_blk, dh, font2,font3,SavePath=results_dir)
            plot_trace(trace1, trace2, trace3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        if (scale.get()) == 5:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            # set the target SNR
            target_SNR_dB = 5

            # compute power of image values
            image_power = images ** 2

            # compute average image and convert to dB
            avg_image_power = np.mean(image_power)
            avg_image_dB = 10 * np.log10(avg_image_power)

            # compute noise and convert to  dB
            avg_noiseImg_dB = avg_image_dB - target_SNR_dB
            avg_noiseImg_power = 10 ** (avg_noiseImg_dB / 10)

            # generate sample of white noise
            mean_noise = 0
            noiseImg2 = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))

            images = images + noiseImg2
            images = torch.from_numpy(images)
            images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
            images = images.to(torch.float32)

            seis = images.view(Inchannels, data_dsp_dim[0], data_dsp_dim[1])
            seis1 = seis.numpy()
            seis   = seis[1,:,:].numpy()

            trace1 = seis1[1,:,50]
            trace2 = seis1[1,:,150]
            trace3 = seis1[1,:,250]

            DisplayOriginalSeismic(seis, data_dsp_dim, data_dsp_blk, dh, font2,font3,SavePath=results_dir)
            plot_trace(trace1, trace2, trace3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        if (scale.get()) == 10:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            # set the target SNR
            target_SNR_dB = 10

            # compute power of image values
            image_power = images ** 2

            # compute average image and convert to dB
            avg_image_power = np.mean(image_power)
            avg_image_dB = 10 * np.log10(avg_image_power)

            # compute noise and convert to  dB
            avg_noiseImg_dB = avg_image_dB - target_SNR_dB
            avg_noiseImg_power = 10 ** (avg_noiseImg_dB / 10)

            # generate sample of white noise
            mean_noise = 0
            noiseImg3 = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))

            images = images + noiseImg3
            images = torch.from_numpy(images)
            images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
            images = images.to(torch.float32)

            seis = images.view(Inchannels, data_dsp_dim[0], data_dsp_dim[1])
            seis1 = seis.numpy()
            seis   = seis[1,:,:].numpy()

            trace1 = seis1[1,:,50]
            trace2 = seis1[1,:,150]
            trace3 = seis1[1,:,250]

            DisplayOriginalSeismic(seis, data_dsp_dim, data_dsp_blk, dh, font2,font3,SavePath=results_dir)
            plot_trace(trace1, trace2, trace3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        if (scale.get()) == 15:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            # set the target SNR
            target_SNR_dB = 15

            # compute power of image values
            image_power = images ** 2

            # compute average image and convert to dB
            avg_image_power = np.mean(image_power)
            avg_image_dB = 10 * np.log10(avg_image_power)

            # compute noise and convert to  dB
            avg_noiseImg_dB = avg_image_dB - target_SNR_dB
            avg_noiseImg_power = 10 ** (avg_noiseImg_dB / 10)

            # generate sample of white noise
            mean_noise = 0
            noiseImg4 = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))

            images = images + noiseImg4
            images = torch.from_numpy(images)
            images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
            images = images.to(torch.float32)

            seis = images.view(Inchannels, data_dsp_dim[0], data_dsp_dim[1])
            seis1 = seis.numpy()
            seis   = seis[1,:,:].numpy()

            trace1 = seis1[1,:,50]
            trace2 = seis1[1,:,150]
            trace3 = seis1[1,:,250]

            DisplayOriginalSeismic(seis, data_dsp_dim, data_dsp_blk, dh, font2,font3,SavePath=results_dir)
            plot_trace(trace1, trace2, trace3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        if (scale.get()) == 20:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            # set the target SNR
            target_SNR_dB = 20

            # compute power of image values
            image_power = images ** 2

            # compute average image and convert to dB
            avg_image_power = np.mean(image_power)
            avg_image_dB = 10 * np.log10(avg_image_power)

            # compute noise and convert to  dB
            avg_noiseImg_dB = avg_image_dB - target_SNR_dB
            avg_noiseImg_power = 10 ** (avg_noiseImg_dB / 10)

            # generate sample of white noise
            mean_noise = 0
            noiseImg5 = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))

            images = images + noiseImg5
            images = torch.from_numpy(images)
            images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
            images = images.to(torch.float32)

            seis = images.view(Inchannels, data_dsp_dim[0], data_dsp_dim[1])
            seis1 = seis.numpy()
            seis   = seis[1,:,:].numpy()

            trace1 = seis1[1,:,50]
            trace2 = seis1[1,:,150]
            trace3 = seis1[1,:,250]

            DisplayOriginalSeismic(seis, data_dsp_dim, data_dsp_blk, dh, font2,font3,SavePath=results_dir)
            plot_trace(trace1, trace2, trace3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        if (scale.get()) == 25:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            # set the target SNR
            target_SNR_dB = 25

            # compute power of image values
            image_power = images ** 2

            # compute average image and convert to dB
            avg_image_power = np.mean(image_power)
            avg_image_dB = 10 * np.log10(avg_image_power)

            # compute noise and convert to  dB
            avg_noiseImg_dB = avg_image_dB - target_SNR_dB
            avg_noiseImg_power = 10 ** (avg_noiseImg_dB / 10)

            # generate sample of white noise
            mean_noise = 0
            noiseImg6 = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))

            images = images + noiseImg6
            images = torch.from_numpy(images)
            images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
            images = images.to(torch.float32)

            seis = images.view(Inchannels, data_dsp_dim[0], data_dsp_dim[1])
            seis1 = seis.numpy()
            seis   = seis[1,:,:].numpy()

            trace1 = seis1[1,:,50]
            trace2 = seis1[1,:,150]
            trace3 = seis1[1,:,250]

            DisplayOriginalSeismic(seis, data_dsp_dim, data_dsp_blk, dh, font2,font3,SavePath=results_dir)
            plot_trace(trace1, trace2, trace3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        if (scale.get()) == 30:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            # set the target SNR
            target_SNR_dB = 30

            # compute power of image values
            image_power = images ** 2

            # compute average image and convert to dB
            avg_image_power = np.mean(image_power)
            avg_image_dB = 10 * np.log10(avg_image_power)

            # compute noise and convert to  dB
            avg_noiseImg_dB = avg_image_dB - target_SNR_dB
            avg_noiseImg_power = 10 ** (avg_noiseImg_dB / 10)

            # generate sample of white noise
            mean_noise = 0
            noiseImg7 = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))

            images = images + noiseImg7
            images = torch.from_numpy(images)
            images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
            images = images.to(torch.float32)

            seis = images.view(Inchannels, data_dsp_dim[0], data_dsp_dim[1])
            seis1 = seis.numpy()
            seis   = seis[1,:,:].numpy()

            trace1 = seis1[1,:,50]
            trace2 = seis1[1,:,150]
            trace3 = seis1[1,:,250]

            DisplayOriginalSeismic(seis, data_dsp_dim, data_dsp_blk, dh, font2,font3,SavePath=results_dir)
            plot_trace(trace1, trace2, trace3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

def seis_with_annot():

    font2 = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 17,
        }
    font3 = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 21,
        }

    images, labels = iter(test_loader).next()
    images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
    labels = labels.view(TestBatchSize,Nclasses,label_dsp_dim[0],label_dsp_dim[1])

    if check_1.get() == 1:

        if (scale.get()) == 0:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            # generate sample of white noise
            images = images + noiseImg1
            images = torch.from_numpy(images)
            images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
            images = images.to(torch.float32)

            seis = images.view(Inchannels, data_dsp_dim[0], data_dsp_dim[1])
            seis1 = seis.numpy()
            seis   = seis[1,:,:].numpy()

            seis_rec1 = int(var4.get())
            seis_rec2 = int(var5.get()) 
            seis_rec3 = int(var6.get())

            trace1 = seis1[1,:,seis_rec1]
            trace2 = seis1[1,:,seis_rec2]
            trace3 = seis1[1,:,seis_rec3]

            display_seismic(seis, seis_rec1, seis_rec2, seis_rec3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)
            plot_trace(trace1, trace2, trace3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        if (scale.get()) == 5:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            # generate sample of white noise
            images = images + noiseImg2
            images = torch.from_numpy(images)
            images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
            images = images.to(torch.float32)

            seis = images.view(Inchannels, data_dsp_dim[0], data_dsp_dim[1])
            seis1 = seis.numpy()
            seis   = seis[1,:,:].numpy()

            seis_rec1 = int(var4.get())
            seis_rec2 = int(var5.get()) 
            seis_rec3 = int(var6.get())

            trace1 = seis1[1,:,seis_rec1]
            trace2 = seis1[1,:,seis_rec2]
            trace3 = seis1[1,:,seis_rec3]

            display_seismic(seis, seis_rec1, seis_rec2, seis_rec3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)
            plot_trace(trace1, trace2, trace3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        if (scale.get()) == 10:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            # generate sample of white noise
            images = images + noiseImg3
            images = torch.from_numpy(images)
            images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
            images = images.to(torch.float32)

            seis = images.view(Inchannels, data_dsp_dim[0], data_dsp_dim[1])
            seis1 = seis.numpy()
            seis   = seis[1,:,:].numpy()

            seis_rec1 = int(var4.get())
            seis_rec2 = int(var5.get()) 
            seis_rec3 = int(var6.get())

            trace1 = seis1[1,:,seis_rec1]
            trace2 = seis1[1,:,seis_rec2]
            trace3 = seis1[1,:,seis_rec3]

            display_seismic(seis, seis_rec1, seis_rec2, seis_rec3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)
            plot_trace(trace1, trace2, trace3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        if (scale.get()) == 15:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            # generate sample of white noise
            images = images + noiseImg4
            images = torch.from_numpy(images)
            images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
            images = images.to(torch.float32)

            seis = images.view(Inchannels, data_dsp_dim[0], data_dsp_dim[1])
            seis1 = seis.numpy()
            seis   = seis[1,:,:].numpy()

            seis_rec1 = int(var4.get())
            seis_rec2 = int(var5.get()) 
            seis_rec3 = int(var6.get())

            trace1 = seis1[1,:,seis_rec1]
            trace2 = seis1[1,:,seis_rec2]
            trace3 = seis1[1,:,seis_rec3]

            display_seismic(seis, seis_rec1, seis_rec2, seis_rec3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)
            plot_trace(trace1, trace2, trace3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        if (scale.get()) == 20:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            # generate sample of white noise
            images = images + noiseImg5
            images = torch.from_numpy(images)
            images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
            images = images.to(torch.float32)

            seis = images.view(Inchannels, data_dsp_dim[0], data_dsp_dim[1])
            seis1 = seis.numpy()
            seis   = seis[1,:,:].numpy()

            seis_rec1 = int(var4.get())
            seis_rec2 = int(var5.get()) 
            seis_rec3 = int(var6.get())

            trace1 = seis1[1,:,seis_rec1]
            trace2 = seis1[1,:,seis_rec2]
            trace3 = seis1[1,:,seis_rec3]

            display_seismic(seis, seis_rec1, seis_rec2, seis_rec3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)
            plot_trace(trace1, trace2, trace3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        if (scale.get()) == 25:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            # generate sample of white noise
            images = images + noiseImg6
            images = torch.from_numpy(images)
            images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
            images = images.to(torch.float32)

            seis = images.view(Inchannels, data_dsp_dim[0], data_dsp_dim[1])
            seis1 = seis.numpy()
            seis   = seis[1,:,:].numpy()

            seis_rec1 = int(var4.get())
            seis_rec2 = int(var5.get()) 
            seis_rec3 = int(var6.get())

            trace1 = seis1[1,:,seis_rec1]
            trace2 = seis1[1,:,seis_rec2]
            trace3 = seis1[1,:,seis_rec3]

            display_seismic(seis, seis_rec1, seis_rec2, seis_rec3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)
            plot_trace(trace1, trace2, trace3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        if (scale.get()) == 30:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            # generate sample of white noise
            images = images + noiseImg7
            images = torch.from_numpy(images)
            images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
            images = images.to(torch.float32)

            seis = images.view(Inchannels, data_dsp_dim[0], data_dsp_dim[1])
            seis1 = seis.numpy()
            seis   = seis[1,:,:].numpy()

            seis_rec1 = int(var4.get())
            seis_rec2 = int(var5.get()) 
            seis_rec3 = int(var6.get())

            trace1 = seis1[1,:,seis_rec1]
            trace2 = seis1[1,:,seis_rec2]
            trace3 = seis1[1,:,seis_rec3]

            display_seismic(seis, seis_rec1, seis_rec2, seis_rec3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)
            plot_trace(trace1, trace2, trace3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

def orig_no_noise_pred(scale_value=0):
    global psnr, ssim, time_elapsed

    since      = time.time()
    TotPSNR    = np.zeros((1,TestSize),dtype=float) 
    TotSSIM    = np.zeros((1,TestSize),dtype=float) 
    Prediction = np.zeros((TestSize,label_dsp_dim[0],label_dsp_dim[1]),dtype=float)
    GT         = np.zeros((TestSize,label_dsp_dim[0],label_dsp_dim[1]),dtype=float)
    Seismic    = np.zeros((TestSize,data_dsp_dim[0],data_dsp_dim[1]),dtype=float) 
    total      = 0

    # Plot one prediction and ground truth
    num = 0
    if SimulateData:
        minvalue = 2000
    else:
        minvalue = 1500
    maxvalue = 4500
    font2 = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 17,
        }
    font3 = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 21,
        }

    images, labels = iter(test_loader).next()
    images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
    labels = labels.view(TestBatchSize,Nclasses,label_dsp_dim[0],label_dsp_dim[1])

    if (check_2.get()) == 0:

        if (scale.get()) == 0:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            for i in range(TestBatchSize):

                # Add noise to the original image
                images = images + noiseImg1
                images = torch.from_numpy(images)
                images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
                images = images.to(torch.float32)

                # Predictions
                net.eval() 
                outputs  = net(images,label_dsp_dim)
                outputs  = outputs.view(TestBatchSize,label_dsp_dim[0],label_dsp_dim[1])
                outputs  = outputs.data.cpu().numpy()
                gts      = labels.data.cpu().numpy()
                outputs1 = outputs

                # Calculate the PSNR, SSIM
                for k in range(TestBatchSize):
                        pd   = outputs[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                        gt   = gts[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])

                        prof1 = outputs1[0,:,50]
                        prof2 = outputs1[0,:,150]
                        prof3 = outputs1[0,:,250]

                        pd   = turn(pd)
                        gt   = turn(gt)

                        Prediction[i*TestBatchSize+k,:,:] = pd
                        GT[i*TestBatchSize+k,:,:] = gt
                        psnr = PSNR(pd,gt)
                        TotPSNR[0,total] = psnr
                        ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                        TotSSIM[0,total] = ssim

            PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            PlotOriginalPrediction(Prediction[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        if (scale.get()) == 5:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            for i in range(TestBatchSize):

                # Add noise to the original image
                images = images + noiseImg2
                images = torch.from_numpy(images)
                images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
                images = images.to(torch.float32)

                # Predictions
                net.eval() 
                outputs  = net(images,label_dsp_dim)
                outputs  = outputs.view(TestBatchSize,label_dsp_dim[0],label_dsp_dim[1])
                outputs  = outputs.data.cpu().numpy()
                gts      = labels.data.cpu().numpy()
                outputs1 = outputs

                # Calculate the PSNR, SSIM
                for k in range(TestBatchSize):
                    pd   = outputs[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                    gt   = gts[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])

                    prof1 = outputs1[0,:,50]
                    prof2 = outputs1[0,:,150]
                    prof3 = outputs1[0,:,250]

                    pd   = turn(pd)
                    gt   = turn(gt)

                    Prediction[i*TestBatchSize+k,:,:] = pd
                    GT[i*TestBatchSize+k,:,:] = gt
                    psnr = PSNR(pd,gt)
                    TotPSNR[0,total] = psnr
                    ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                    TotSSIM[0,total] = ssim

            PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            PlotOriginalPrediction(Prediction[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        elif (scale.get()) == 10:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            for i in range(TestBatchSize):

                # Add noise to the original image
                images = images + noiseImg3
                images = torch.from_numpy(images)
                images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
                images = images.to(torch.float32)

                # Predictions
                net.eval() 
                outputs  = net(images,label_dsp_dim)
                outputs  = outputs.view(TestBatchSize,label_dsp_dim[0],label_dsp_dim[1])
                outputs  = outputs.data.cpu().numpy()
                gts      = labels.data.cpu().numpy()
                outputs1 = outputs

                # Calculate the PSNR, SSIM
                for k in range(TestBatchSize):
                    pd   = outputs[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                    gt   = gts[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])

                    prof1 = outputs1[0,:,50]
                    prof2 = outputs1[0,:,150]
                    prof3 = outputs1[0,:,250]

                    pd   = turn(pd)
                    gt   = turn(gt)

                    Prediction[i*TestBatchSize+k,:,:] = pd
                    GT[i*TestBatchSize+k,:,:] = gt
                    psnr = PSNR(pd,gt)
                    TotPSNR[0,total] = psnr
                    ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                    TotSSIM[0,total] = ssim

            PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            PlotOriginalPrediction(Prediction[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        if (scale.get()) == 15:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            for i in range(TestBatchSize):

                # Add noise to the original image
                images = images + noiseImg4
                images = torch.from_numpy(images)
                images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
                images = images.to(torch.float32)

                # Predictions
                net.eval() 
                outputs  = net(images,label_dsp_dim)
                outputs  = outputs.view(TestBatchSize,label_dsp_dim[0],label_dsp_dim[1])
                outputs  = outputs.data.cpu().numpy()
                gts      = labels.data.cpu().numpy()
                outputs1 = outputs

                # Calculate the PSNR, SSIM
                for k in range(TestBatchSize):
                    pd   = outputs[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                    gt   = gts[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])

                    prof1 = outputs1[0,:,50]
                    prof2 = outputs1[0,:,150]
                    prof3 = outputs1[0,:,250]

                    pd   = turn(pd)
                    gt   = turn(gt)

                    Prediction[i*TestBatchSize+k,:,:] = pd
                    GT[i*TestBatchSize+k,:,:] = gt
                    psnr = PSNR(pd,gt)
                    TotPSNR[0,total] = psnr
                    ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                    TotSSIM[0,total] = ssim

            PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            PlotOriginalPrediction(Prediction[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        elif (scale.get()) == 20:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            for i in range(TestBatchSize):

                # Add noise to the original image
                images = images + noiseImg5
                images = torch.from_numpy(images)
                images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
                images = images.to(torch.float32)

                # Predictions
                net.eval() 
                outputs  = net(images,label_dsp_dim)
                outputs  = outputs.view(TestBatchSize,label_dsp_dim[0],label_dsp_dim[1])
                outputs  = outputs.data.cpu().numpy()
                gts      = labels.data.cpu().numpy()
                outputs1 = outputs

                # Calculate the PSNR, SSIM
                for k in range(TestBatchSize):
                    pd   = outputs[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                    gt   = gts[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])

                    prof1 = outputs1[0,:,50]
                    prof2 = outputs1[0,:,150]
                    prof3 = outputs1[0,:,250]

                    pd   = turn(pd)
                    gt   = turn(gt)

                    Prediction[i*TestBatchSize+k,:,:] = pd
                    GT[i*TestBatchSize+k,:,:] = gt
                    psnr = PSNR(pd,gt)
                    TotPSNR[0,total] = psnr
                    ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                    TotSSIM[0,total] = ssim

            PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            PlotOriginalPrediction(Prediction[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        if (scale.get()) == 25:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            for i in range(TestBatchSize):

                # Add noise to the original image
                images = images + noiseImg6
                images = torch.from_numpy(images)
                images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
                images = images.to(torch.float32)

                # Predictions
                net.eval() 
                outputs  = net(images,label_dsp_dim)
                outputs  = outputs.view(TestBatchSize,label_dsp_dim[0],label_dsp_dim[1])
                outputs  = outputs.data.cpu().numpy()
                gts      = labels.data.cpu().numpy()
                outputs1 = outputs

                # Calculate the PSNR, SSIM
                for k in range(TestBatchSize):
                    pd   = outputs[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                    gt   = gts[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])

                    prof1 = outputs1[0,:,50]
                    prof2 = outputs1[0,:,150]
                    prof3 = outputs1[0,:,250]

                    pd   = turn(pd)
                    gt   = turn(gt)

                    Prediction[i*TestBatchSize+k,:,:] = pd
                    GT[i*TestBatchSize+k,:,:] = gt
                    psnr = PSNR(pd,gt)
                    TotPSNR[0,total] = psnr
                    ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                    TotSSIM[0,total] = ssim

            PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            PlotOriginalPrediction(Prediction[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        if (scale.get()) == 30:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            for i in range(TestBatchSize):

                # Add noise to the original image
                images = images + noiseImg7
                images = torch.from_numpy(images)
                images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
                images = images.to(torch.float32)

                # Predictions
                net.eval() 
                outputs  = net(images,label_dsp_dim)
                outputs  = outputs.view(TestBatchSize,label_dsp_dim[0],label_dsp_dim[1])
                outputs  = outputs.data.cpu().numpy()
                gts      = labels.data.cpu().numpy()
                outputs1 = outputs

                # Calculate the PSNR, SSIM
                for k in range(TestBatchSize):
                    pd   = outputs[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                    gt   = gts[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])

                    prof1 = outputs1[0,:,50]
                    prof2 = outputs1[0,:,150]
                    prof3 = outputs1[0,:,250]

                    pd   = turn(pd)
                    gt   = turn(gt)

                    Prediction[i*TestBatchSize+k,:,:] = pd
                    GT[i*TestBatchSize+k,:,:] = gt
                    psnr = PSNR(pd,gt)
                    TotPSNR[0,total] = psnr
                    ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                    TotSSIM[0,total] = ssim

            PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            PlotOriginalPrediction(Prediction[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        SaveTestResults(TotPSNR,TotSSIM,Prediction,GT,results_dir)

    # Record the consuming time
    time_elapsed = time.time() - since

    screen.insert(INSERT, " " + '\n')
    screen.insert(INSERT, "No-noise UNet Model Loaded ...." + '\n')
    screen.insert(INSERT, " " + '\n')
    
    return psnr, ssim, time_elapsed

net_noise = FCNVMB_test.net_noise

def orig_noise_aware_pred(scale_value=0):
    global psnr, ssim, time_elapsed

    since      = time.time()
    TotPSNR    = np.zeros((1,TestSize),dtype=float) 
    TotSSIM    = np.zeros((1,TestSize),dtype=float) 
    Prediction = np.zeros((TestSize,label_dsp_dim[0],label_dsp_dim[1]),dtype=float)
    GT         = np.zeros((TestSize,label_dsp_dim[0],label_dsp_dim[1]),dtype=float)
    Seismic    = np.zeros((TestSize,data_dsp_dim[0],data_dsp_dim[1]),dtype=float) 
    total      = 0

    # Plot one prediction and ground truth
    num = 0
    if SimulateData:
        minvalue = 2000
    else:
        minvalue = 1500
    maxvalue = 4500
    font2 = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 17,
        }
    font3 = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 21,
        }

    images, labels = iter(test_loader).next()
    images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
    labels = labels.view(TestBatchSize,Nclasses,label_dsp_dim[0],label_dsp_dim[1])

    if (check_3.get()) == 0:

        if (scale.get()) == 0:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            for i in range(TestBatchSize):

                # Add noise to the original image
                images = images + noiseImg1
                images = torch.from_numpy(images)
                images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
                images = images.to(torch.float32)

                # Predictions
                net_noise.eval() 
                outputs  = net_noise(images,label_dsp_dim)
                outputs  = outputs.view(TestBatchSize,label_dsp_dim[0],label_dsp_dim[1])
                outputs  = outputs.data.cpu().numpy()
                gts      = labels.data.cpu().numpy()
                outputs1 = outputs

                # Calculate the PSNR, SSIM
                for k in range(TestBatchSize):
                        pd   = outputs[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                        gt   = gts[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])

                        prof1 = outputs1[0,:,50]
                        prof2 = outputs1[0,:,150]
                        prof3 = outputs1[0,:,250]

                        pd   = turn(pd)
                        gt   = turn(gt)

                        Prediction[i*TestBatchSize+k,:,:] = pd
                        GT[i*TestBatchSize+k,:,:] = gt
                        psnr = PSNR(pd,gt)
                        TotPSNR[0,total] = psnr
                        ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                        TotSSIM[0,total] = ssim

            PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            PlotOriginalPrediction(Prediction[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        if (scale.get()) == 5:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            for i in range(TestBatchSize):

                # Add noise to the original image
                images = images + noiseImg2
                images = torch.from_numpy(images)
                images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
                images = images.to(torch.float32)

                # Predictions
                net_noise.eval() 
                outputs  = net_noise(images,label_dsp_dim)
                outputs  = outputs.view(TestBatchSize,label_dsp_dim[0],label_dsp_dim[1])
                outputs  = outputs.data.cpu().numpy()
                gts      = labels.data.cpu().numpy()
                outputs1 = outputs

                # Calculate the PSNR, SSIM
                for k in range(TestBatchSize):
                    pd   = outputs[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                    gt   = gts[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])

                    prof1 = outputs1[0,:,50]
                    prof2 = outputs1[0,:,150]
                    prof3 = outputs1[0,:,250]

                    pd   = turn(pd)
                    gt   = turn(gt)

                    Prediction[i*TestBatchSize+k,:,:] = pd
                    GT[i*TestBatchSize+k,:,:] = gt
                    psnr = PSNR(pd,gt)
                    TotPSNR[0,total] = psnr
                    ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                    TotSSIM[0,total] = ssim

            PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            PlotOriginalPrediction(Prediction[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        elif (scale.get()) == 10:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            for i in range(TestBatchSize):

                # Add noise to the original image
                images = images + noiseImg3
                images = torch.from_numpy(images)
                images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
                images = images.to(torch.float32)

                # Predictions
                net_noise.eval() 
                outputs  = net_noise(images,label_dsp_dim)
                outputs  = outputs.view(TestBatchSize,label_dsp_dim[0],label_dsp_dim[1])
                outputs  = outputs.data.cpu().numpy()
                gts      = labels.data.cpu().numpy()
                outputs1 = outputs

                # Calculate the PSNR, SSIM
                for k in range(TestBatchSize):
                    pd   = outputs[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                    gt   = gts[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                
                    prof1 = outputs1[0,:,50]
                    prof2 = outputs1[0,:,150]
                    prof3 = outputs1[0,:,250]

                    pd   = turn(pd)
                    gt   = turn(gt)

                    Prediction[i*TestBatchSize+k,:,:] = pd
                    GT[i*TestBatchSize+k,:,:] = gt
                    psnr = PSNR(pd,gt)
                    TotPSNR[0,total] = psnr
                    ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                    TotSSIM[0,total] = ssim

            PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            PlotOriginalPrediction(Prediction[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        if (scale.get()) == 15:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            for i in range(TestBatchSize):

                # Add noise to the original image
                images = images + noiseImg4
                images = torch.from_numpy(images)
                images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
                images = images.to(torch.float32)

                # Predictions
                net_noise.eval() 
                outputs  = net_noise(images,label_dsp_dim)
                outputs  = outputs.view(TestBatchSize,label_dsp_dim[0],label_dsp_dim[1])
                outputs  = outputs.data.cpu().numpy()
                gts      = labels.data.cpu().numpy()
                outputs1 = outputs

                # Calculate the PSNR, SSIM
                for k in range(TestBatchSize):
                    pd   = outputs[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                    gt   = gts[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                
                    prof1 = outputs1[0,:,50]
                    prof2 = outputs1[0,:,150]
                    prof3 = outputs1[0,:,250]

                    pd   = turn(pd)
                    gt   = turn(gt)

                    Prediction[i*TestBatchSize+k,:,:] = pd
                    GT[i*TestBatchSize+k,:,:] = gt
                    psnr = PSNR(pd,gt)
                    TotPSNR[0,total] = psnr
                    ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                    TotSSIM[0,total] = ssim

            PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            PlotOriginalPrediction(Prediction[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        elif (scale.get()) == 20:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            for i in range(TestBatchSize):

                # Add noise to the original image
                images = images + noiseImg5
                images = torch.from_numpy(images)
                images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
                images = images.to(torch.float32)

                # Predictions
                net_noise.eval() 
                outputs  = net_noise(images,label_dsp_dim)
                outputs  = outputs.view(TestBatchSize,label_dsp_dim[0],label_dsp_dim[1])
                outputs  = outputs.data.cpu().numpy()
                gts      = labels.data.cpu().numpy()
                outputs1 = outputs

                # Calculate the PSNR, SSIM
                for k in range(TestBatchSize):
                    pd   = outputs[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                    gt   = gts[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                
                    prof1 = outputs1[0,:,50]
                    prof2 = outputs1[0,:,150]
                    prof3 = outputs1[0,:,250]

                    pd   = turn(pd)
                    gt   = turn(gt)

                    Prediction[i*TestBatchSize+k,:,:] = pd
                    GT[i*TestBatchSize+k,:,:] = gt
                    psnr = PSNR(pd,gt)
                    TotPSNR[0,total] = psnr
                    ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                    TotSSIM[0,total] = ssim

            PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            PlotOriginalPrediction(Prediction[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        if (scale.get()) == 25:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            for i in range(TestBatchSize):

                # Add noise to the original image
                images = images + noiseImg6
                images = torch.from_numpy(images)
                images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
                images = images.to(torch.float32)

                # Predictions
                net_noise.eval() 
                outputs  = net_noise(images,label_dsp_dim)
                outputs  = outputs.view(TestBatchSize,label_dsp_dim[0],label_dsp_dim[1])
                outputs  = outputs.data.cpu().numpy()
                gts      = labels.data.cpu().numpy()
                outputs1 = outputs

                # Calculate the PSNR, SSIM
                for k in range(TestBatchSize):
                    pd   = outputs[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                    gt   = gts[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])

                    prof1 = outputs1[0,:,50]
                    prof2 = outputs1[0,:,150]
                    prof3 = outputs1[0,:,250]

                    pd   = turn(pd)
                    gt   = turn(gt)

                    Prediction[i*TestBatchSize+k,:,:] = pd
                    GT[i*TestBatchSize+k,:,:] = gt
                    psnr = PSNR(pd,gt)
                    TotPSNR[0,total] = psnr
                    ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                    TotSSIM[0,total] = ssim

            PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            PlotOriginalPrediction(Prediction[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        if (scale.get()) == 30:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            for i in range(TestBatchSize):

                # Add noise to the original image
                images = images + noiseImg7
                images = torch.from_numpy(images)
                images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
                images = images.to(torch.float32)

                # Predictions
                net_noise.eval() 
                outputs  = net_noise(images,label_dsp_dim)
                outputs  = outputs.view(TestBatchSize,label_dsp_dim[0],label_dsp_dim[1])
                outputs  = outputs.data.cpu().numpy()
                gts      = labels.data.cpu().numpy()
                outputs1 = outputs

                # Calculate the PSNR, SSIM
                for k in range(TestBatchSize):
                    pd   = outputs[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                    gt   = gts[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                
                    prof1 = outputs1[0,:,50]
                    prof2 = outputs1[0,:150]
                    prof3 = outputs1[0,:,250]

                    pd   = turn(pd)
                    gt   = turn(gt)

                    Prediction[i*TestBatchSize+k,:,:] = pd
                    GT[i*TestBatchSize+k,:,:] = gt
                    psnr = PSNR(pd,gt)
                    TotPSNR[0,total] = psnr
                    ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                    TotSSIM[0,total] = ssim

            PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            PlotOriginalPrediction(Prediction[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        SaveTestResults(TotPSNR,TotSSIM,Prediction,GT,results_dir)

    # Record the consuming time
    time_elapsed = time.time() - since

    screen.insert(INSERT, " " + '\n')
    screen.insert(INSERT, "Noise-aware UNet Model Loaded ...." + '\n')
    screen.insert(INSERT, " " + '\n')
    
    return psnr, ssim, time_elapsed

def no_noise(scale_value=0):
    global psnr, ssim, time_elapsed

    since      = time.time()
    TotPSNR    = np.zeros((1,TestSize),dtype=float) 
    TotSSIM    = np.zeros((1,TestSize),dtype=float) 
    Prediction = np.zeros((TestSize,label_dsp_dim[0],label_dsp_dim[1]),dtype=float)
    GT         = np.zeros((TestSize,label_dsp_dim[0],label_dsp_dim[1]),dtype=float)
    Seismic    = np.zeros((TestSize,data_dsp_dim[0],data_dsp_dim[1]),dtype=float) 
    total      = 0

    # Plot one prediction and ground truth
    num = 0
    if SimulateData:
        minvalue = 2000
    else:
        minvalue = 1500
    maxvalue = 4500
    font2 = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 17,
        }
    font3 = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 21,
        }

    images, labels = iter(test_loader).next()
    images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
    labels = labels.view(TestBatchSize,Nclasses,label_dsp_dim[0],label_dsp_dim[1])

    if (check_2.get()) == 1:

        if (scale.get()) == 0:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            for i in range(TestBatchSize):

                # Add noise to the original image
                images = images + noiseImg1
                images = torch.from_numpy(images)
                images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
                images = images.to(torch.float32)

                # Predictions
                net.eval() 
                outputs  = net(images,label_dsp_dim)
                outputs  = outputs.view(TestBatchSize,label_dsp_dim[0],label_dsp_dim[1])
                outputs  = outputs.data.cpu().numpy()
                gts      = labels.data.cpu().numpy()
                outputs1 = outputs

                # Calculate the PSNR, SSIM
                for k in range(TestBatchSize):
                        pd   = outputs[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                        gt   = gts[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                    
                        vel_rec1 = float(var7.get())
                        vel_rec2 = float(var8.get())
                        vel_rec3 = float(var9.get())

                        vel_rec1 = (vel_rec1 * 100)
                        vel_rec2 = (vel_rec2 * 100)
                        vel_rec3 = (vel_rec3 * 100)
                        
                        frac1, whole1 = math.modf(vel_rec1)
                        frac2, whole2 = math.modf(vel_rec2)
                        frac3, whole3 = math.modf(vel_rec3)

                        if frac1 >= 0.5:
                            vel_rec1 = math.ceil(vel_rec1)
                        else:
                            vel_rec1 = round(vel_rec1)

                        if frac2 >= 0.5:
                            vel_rec2 = math.ceil(vel_rec2)
                        else:
                            vel_rec2 = round(vel_rec2)

                        if frac3 >= 0.5:
                            vel_rec3 = math.ceil(vel_rec3)
                        else:
                            vel_rec3 = round(vel_rec3)    

                        vel_rec1 = int(vel_rec1)
                        vel_rec2 = int(vel_rec2)
                        vel_rec3 = int(vel_rec3)

                        prof1 = outputs1[0,:,vel_rec1]
                        prof2 = outputs1[0,:,vel_rec2]
                        prof3 = outputs1[0,:,vel_rec3]

                        pd   = turn(pd)
                        gt   = turn(gt)

                        Prediction[i*TestBatchSize+k,:,:] = pd
                        GT[i*TestBatchSize+k,:,:] = gt
                        psnr = PSNR(pd,gt)
                        TotPSNR[0,total] = psnr
                        ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                        TotSSIM[0,total] = ssim

            PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            PlotPrediction(Prediction[num,:,:],vel_rec1, vel_rec2, vel_rec3,label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        if (scale.get()) == 5:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            for i in range(TestBatchSize):

                # Add noise to the original image
                images = images + noiseImg2
                images = torch.from_numpy(images)
                images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
                images = images.to(torch.float32)

                # Predictions
                net.eval() 
                outputs  = net(images,label_dsp_dim)
                outputs  = outputs.view(TestBatchSize,label_dsp_dim[0],label_dsp_dim[1])
                outputs  = outputs.data.cpu().numpy()
                gts      = labels.data.cpu().numpy()
                outputs1 = outputs

                # Calculate the PSNR, SSIM
                for k in range(TestBatchSize):
                    pd   = outputs[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                    gt   = gts[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])

                    vel_rec1 = float(var7.get())
                    vel_rec2 = float(var8.get())
                    vel_rec3 = float(var9.get())

                    vel_rec1 = (vel_rec1 * 100)
                    vel_rec2 = (vel_rec2 * 100)
                    vel_rec3 = (vel_rec3 * 100)
                    
                    frac1, whole1 = math.modf(vel_rec1)
                    frac2, whole2 = math.modf(vel_rec2)
                    frac3, whole3 = math.modf(vel_rec3)

                    if frac1 >= 0.5:
                        vel_rec1 = math.ceil(vel_rec1)
                    else:
                        vel_rec1 = round(vel_rec1)

                    if frac2 >= 0.5:
                        vel_rec2 = math.ceil(vel_rec2)
                    else:
                        vel_rec2 = round(vel_rec2)

                    if frac3 >= 0.5:
                        vel_rec3 = math.ceil(vel_rec3)
                    else:
                        vel_rec3 = round(vel_rec3)    

                    vel_rec1 = int(vel_rec1)
                    vel_rec2 = int(vel_rec2)
                    vel_rec3 = int(vel_rec3)

                    prof1 = outputs1[0,:,vel_rec1]
                    prof2 = outputs1[0,:,vel_rec2]
                    prof3 = outputs1[0,:,vel_rec3]

                    pd   = turn(pd)
                    gt   = turn(gt)

                    Prediction[i*TestBatchSize+k,:,:] = pd
                    GT[i*TestBatchSize+k,:,:] = gt
                    psnr = PSNR(pd,gt)
                    TotPSNR[0,total] = psnr
                    ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                    TotSSIM[0,total] = ssim

            PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            PlotPrediction(Prediction[num,:,:],vel_rec1, vel_rec2, vel_rec3,label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        elif (scale.get()) == 10:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            for i in range(TestBatchSize):

                # Add noise to the original image
                images = images + noiseImg3
                images = torch.from_numpy(images)
                images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
                images = images.to(torch.float32)

                # Predictions
                net.eval() 
                outputs  = net(images,label_dsp_dim)
                outputs  = outputs.view(TestBatchSize,label_dsp_dim[0],label_dsp_dim[1])
                outputs  = outputs.data.cpu().numpy()
                gts      = labels.data.cpu().numpy()
                outputs1 = outputs

                # Calculate the PSNR, SSIM
                for k in range(TestBatchSize):
                    pd   = outputs[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                    gt   = gts[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                
                    vel_rec1 = float(var7.get())
                    vel_rec2 = float(var8.get())
                    vel_rec3 = float(var9.get())

                    vel_rec1 = (vel_rec1 * 100)
                    vel_rec2 = (vel_rec2 * 100)
                    vel_rec3 = (vel_rec3 * 100)
                    
                    frac1, whole1 = math.modf(vel_rec1)
                    frac2, whole2 = math.modf(vel_rec2)
                    frac3, whole3 = math.modf(vel_rec3)

                    if frac1 >= 0.5:
                        vel_rec1 = math.ceil(vel_rec1)
                    else:
                        vel_rec1 = round(vel_rec1)

                    if frac2 >= 0.5:
                        vel_rec2 = math.ceil(vel_rec2)
                    else:
                        vel_rec2 = round(vel_rec2)

                    if frac3 >= 0.5:
                        vel_rec3 = math.ceil(vel_rec3)
                    else:
                        vel_rec3 = round(vel_rec3)    

                    vel_rec1 = int(vel_rec1)
                    vel_rec2 = int(vel_rec2)
                    vel_rec3 = int(vel_rec3)

                    prof1 = outputs1[0,:,vel_rec1]
                    prof2 = outputs1[0,:,vel_rec2]
                    prof3 = outputs1[0,:,vel_rec3]

                    pd   = turn(pd)
                    gt   = turn(gt)

                    Prediction[i*TestBatchSize+k,:,:] = pd
                    GT[i*TestBatchSize+k,:,:] = gt
                    psnr = PSNR(pd,gt)
                    TotPSNR[0,total] = psnr
                    ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                    TotSSIM[0,total] = ssim

            PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            PlotPrediction(Prediction[num,:,:],vel_rec1, vel_rec2, vel_rec3,label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        if (scale.get()) == 15:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            for i in range(TestBatchSize):

                # Add noise to the original image
                images = images + noiseImg4
                images = torch.from_numpy(images)
                images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
                images = images.to(torch.float32)

                # Predictions
                net.eval() 
                outputs  = net(images,label_dsp_dim)
                outputs  = outputs.view(TestBatchSize,label_dsp_dim[0],label_dsp_dim[1])
                outputs  = outputs.data.cpu().numpy()
                gts      = labels.data.cpu().numpy()
                outputs1 = outputs

                # Calculate the PSNR, SSIM
                for k in range(TestBatchSize):
                    pd   = outputs[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                    gt   = gts[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                
                    vel_rec1 = float(var7.get())
                    vel_rec2 = float(var8.get())
                    vel_rec3 = float(var9.get())

                    vel_rec1 = (vel_rec1 * 100)
                    vel_rec2 = (vel_rec2 * 100)
                    vel_rec3 = (vel_rec3 * 100)
                    
                    frac1, whole1 = math.modf(vel_rec1)
                    frac2, whole2 = math.modf(vel_rec2)
                    frac3, whole3 = math.modf(vel_rec3)

                    if frac1 >= 0.5:
                        vel_rec1 = math.ceil(vel_rec1)
                    else:
                        vel_rec1 = round(vel_rec1)

                    if frac2 >= 0.5:
                        vel_rec2 = math.ceil(vel_rec2)
                    else:
                        vel_rec2 = round(vel_rec2)

                    if frac3 >= 0.5:
                        vel_rec3 = math.ceil(vel_rec3)
                    else:
                        vel_rec3 = round(vel_rec3)    

                    vel_rec1 = int(vel_rec1)
                    vel_rec2 = int(vel_rec2)
                    vel_rec3 = int(vel_rec3)

                    prof1 = outputs1[0,:,vel_rec1]
                    prof2 = outputs1[0,:,vel_rec2]
                    prof3 = outputs1[0,:,vel_rec3]

                    pd   = turn(pd)
                    gt   = turn(gt)

                    Prediction[i*TestBatchSize+k,:,:] = pd
                    GT[i*TestBatchSize+k,:,:] = gt
                    psnr = PSNR(pd,gt)
                    TotPSNR[0,total] = psnr
                    ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                    TotSSIM[0,total] = ssim

            PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            PlotPrediction(Prediction[num,:,:],vel_rec1, vel_rec2, vel_rec3,label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        elif (scale.get()) == 20:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)
        
            for i in range(TestBatchSize):

                # Add noise to the original image
                images = images + noiseImg5
                images = torch.from_numpy(images)
                images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
                images = images.to(torch.float32)

                # Predictions
                net.eval() 
                outputs  = net(images,label_dsp_dim)
                outputs  = outputs.view(TestBatchSize,label_dsp_dim[0],label_dsp_dim[1])
                outputs  = outputs.data.cpu().numpy()
                gts      = labels.data.cpu().numpy()
                outputs1 = outputs

                # Calculate the PSNR, SSIM
                for k in range(TestBatchSize):
                    pd   = outputs[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                    gt   = gts[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                
                    vel_rec1 = float(var7.get())
                    vel_rec2 = float(var8.get())
                    vel_rec3 = float(var9.get())

                    vel_rec1 = (vel_rec1 * 100)
                    vel_rec2 = (vel_rec2 * 100)
                    vel_rec3 = (vel_rec3 * 100)
                    
                    frac1, whole1 = math.modf(vel_rec1)
                    frac2, whole2 = math.modf(vel_rec2)
                    frac3, whole3 = math.modf(vel_rec3)

                    if frac1 >= 0.5:
                        vel_rec1 = math.ceil(vel_rec1)
                    else:
                        vel_rec1 = round(vel_rec1)

                    if frac2 >= 0.5:
                        vel_rec2 = math.ceil(vel_rec2)
                    else:
                        vel_rec2 = round(vel_rec2)

                    if frac3 >= 0.5:
                        vel_rec3 = math.ceil(vel_rec3)
                    else:
                        vel_rec3 = round(vel_rec3)    

                    vel_rec1 = int(vel_rec1)
                    vel_rec2 = int(vel_rec2)
                    vel_rec3 = int(vel_rec3)

                    prof1 = outputs1[0,:,vel_rec1]
                    prof2 = outputs1[0,:,vel_rec2]
                    prof3 = outputs1[0,:,vel_rec3]

                    pd   = turn(pd)
                    gt   = turn(gt)

                    Prediction[i*TestBatchSize+k,:,:] = pd
                    GT[i*TestBatchSize+k,:,:] = gt
                    psnr = PSNR(pd,gt)
                    TotPSNR[0,total] = psnr
                    ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                    TotSSIM[0,total] = ssim

            PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            PlotPrediction(Prediction[num,:,:],vel_rec1, vel_rec2, vel_rec3,label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        if (scale.get()) == 25:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            for i in range(TestBatchSize):

                # Add noise to the original image
                images = images + noiseImg6
                images = torch.from_numpy(images)
                images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
                images = images.to(torch.float32)

                # Predictions
                net.eval() 
                outputs  = net(images,label_dsp_dim)
                outputs  = outputs.view(TestBatchSize,label_dsp_dim[0],label_dsp_dim[1])
                outputs  = outputs.data.cpu().numpy()
                gts      = labels.data.cpu().numpy()
                outputs1 = outputs

                # Calculate the PSNR, SSIM
                for k in range(TestBatchSize):
                    pd   = outputs[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                    gt   = gts[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])

                    vel_rec1 = float(var7.get())
                    vel_rec2 = float(var8.get())
                    vel_rec3 = float(var9.get())

                    vel_rec1 = (vel_rec1 * 100)
                    vel_rec2 = (vel_rec2 * 100)
                    vel_rec3 = (vel_rec3 * 100)
                    
                    frac1, whole1 = math.modf(vel_rec1)
                    frac2, whole2 = math.modf(vel_rec2)
                    frac3, whole3 = math.modf(vel_rec3)

                    if frac1 >= 0.5:
                        vel_rec1 = math.ceil(vel_rec1)
                    else:
                        vel_rec1 = round(vel_rec1)

                    if frac2 >= 0.5:
                        vel_rec2 = math.ceil(vel_rec2)
                    else:
                        vel_rec2 = round(vel_rec2)

                    if frac3 >= 0.5:
                        vel_rec3 = math.ceil(vel_rec3)
                    else:
                        vel_rec3 = round(vel_rec3)    

                    vel_rec1 = int(vel_rec1)
                    vel_rec2 = int(vel_rec2)
                    vel_rec3 = int(vel_rec3)

                    prof1 = outputs1[0,:,vel_rec1]
                    prof2 = outputs1[0,:,vel_rec2]
                    prof3 = outputs1[0,:,vel_rec3]

                    pd   = turn(pd)
                    gt   = turn(gt)

                    Prediction[i*TestBatchSize+k,:,:] = pd
                    GT[i*TestBatchSize+k,:,:] = gt
                    psnr = PSNR(pd,gt)
                    TotPSNR[0,total] = psnr
                    ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                    TotSSIM[0,total] = ssim

            PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            PlotPrediction(Prediction[num,:,:],vel_rec1, vel_rec2, vel_rec3,label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        if (scale.get()) == 30:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            for i in range(TestBatchSize):

                # Add noise to the original image
                images = images + noiseImg7
                images = torch.from_numpy(images)
                images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
                images = images.to(torch.float32)

                # Predictions
                net.eval() 
                outputs  = net(images,label_dsp_dim)
                outputs  = outputs.view(TestBatchSize,label_dsp_dim[0],label_dsp_dim[1])
                outputs  = outputs.data.cpu().numpy()
                gts      = labels.data.cpu().numpy()
                outputs1 = outputs

                # Calculate the PSNR, SSIM
                for k in range(TestBatchSize):
                    pd   = outputs[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                    gt   = gts[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])

                    vel_rec1 = float(var7.get())
                    vel_rec2 = float(var8.get())
                    vel_rec3 = float(var9.get())

                    vel_rec1 = (vel_rec1 * 100)
                    vel_rec2 = (vel_rec2 * 100)
                    vel_rec3 = (vel_rec3 * 100)
                    
                    frac1, whole1 = math.modf(vel_rec1)
                    frac2, whole2 = math.modf(vel_rec2)
                    frac3, whole3 = math.modf(vel_rec3)

                    if frac1 >= 0.5:
                        vel_rec1 = math.ceil(vel_rec1)
                    else:
                        vel_rec1 = round(vel_rec1)

                    if frac2 >= 0.5:
                        vel_rec2 = math.ceil(vel_rec2)
                    else:
                        vel_rec2 = round(vel_rec2)

                    if frac3 >= 0.5:
                        vel_rec3 = math.ceil(vel_rec3)
                    else:
                        vel_rec3 = round(vel_rec3)    

                    vel_rec1 = int(vel_rec1)
                    vel_rec2 = int(vel_rec2)
                    vel_rec3 = int(vel_rec3)

                    prof1 = outputs1[0,:,vel_rec1]
                    prof2 = outputs1[0,:,vel_rec2]
                    prof3 = outputs1[0,:,vel_rec3]

                    pd   = turn(pd)
                    gt   = turn(gt)

                    Prediction[i*TestBatchSize+k,:,:] = pd
                    GT[i*TestBatchSize+k,:,:] = gt
                    psnr = PSNR(pd,gt)
                    TotPSNR[0,total] = psnr
                    ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                    TotSSIM[0,total] = ssim

            PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            PlotPrediction(Prediction[num,:,:],vel_rec1, vel_rec2, vel_rec3,label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        SaveTestResults(TotPSNR,TotSSIM,Prediction,GT,results_dir)

        # Record the consuming time
        time_elapsed = time.time() - since

    # screen.insert(INSERT, " " + '\n')
    # screen.insert(INSERT, "No-noise UNet Model Loaded ...." + '\n')
    # screen.insert(INSERT, " " + '\n')
    else:
        pass
    
    return psnr, ssim, time_elapsed

net_noise = FCNVMB_test.net_noise

def noise_aware(scale_value=0):
    global psnr, ssim, time_elapsed

    since      = time.time()
    TotPSNR    = np.zeros((1,TestSize),dtype=float) 
    TotSSIM    = np.zeros((1,TestSize),dtype=float) 
    Prediction = np.zeros((TestSize,label_dsp_dim[0],label_dsp_dim[1]),dtype=float)
    GT         = np.zeros((TestSize,label_dsp_dim[0],label_dsp_dim[1]),dtype=float)
    Seismic    = np.zeros((TestSize,data_dsp_dim[0],data_dsp_dim[1]),dtype=float) 
    total      = 0

    # Plot one prediction and ground truth
    num = 0
    if SimulateData:
        minvalue = 2000
    else:
        minvalue = 1500
    maxvalue = 4500
    font2 = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 17,
        }
    font3 = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 21,
        }

    images, labels = iter(test_loader).next()
    images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
    labels = labels.view(TestBatchSize,Nclasses,label_dsp_dim[0],label_dsp_dim[1])

    if (check_3.get()) == 1:

        if (scale.get()) == 0:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            for i in range(TestBatchSize):

                # Add noise to the original image
                images = images + noiseImg1
                images = torch.from_numpy(images)
                images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
                images = images.to(torch.float32)

                # Predictions
                net_noise.eval() 
                outputs  = net_noise(images,label_dsp_dim)
                outputs  = outputs.view(TestBatchSize,label_dsp_dim[0],label_dsp_dim[1])
                outputs  = outputs.data.cpu().numpy()
                gts      = labels.data.cpu().numpy()
                outputs1 = outputs

                # Calculate the PSNR, SSIM
                for k in range(TestBatchSize):
                        pd   = outputs[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                        gt   = gts[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                    
                        vel_rec1 = float(var7.get())
                        vel_rec2 = float(var8.get())
                        vel_rec3 = float(var9.get())

                        vel_rec1 = (vel_rec1 * 100)
                        vel_rec2 = (vel_rec2 * 100)
                        vel_rec3 = (vel_rec3 * 100)
                        
                        frac1, whole1 = math.modf(vel_rec1)
                        frac2, whole2 = math.modf(vel_rec2)
                        frac3, whole3 = math.modf(vel_rec3)

                        if frac1 >= 0.5:
                            vel_rec1 = math.ceil(vel_rec1)
                        else:
                            vel_rec1 = round(vel_rec1)

                        if frac2 >= 0.5:
                            vel_rec2 = math.ceil(vel_rec2)
                        else:
                            vel_rec2 = round(vel_rec2)

                        if frac3 >= 0.5:
                            vel_rec3 = math.ceil(vel_rec3)
                        else:
                            vel_rec3 = round(vel_rec3)    

                        vel_rec1 = int(vel_rec1)
                        vel_rec2 = int(vel_rec2)
                        vel_rec3 = int(vel_rec3)

                        prof1 = outputs1[0,:,vel_rec1]
                        prof2 = outputs1[0,:,vel_rec2]
                        prof3 = outputs1[0,:,vel_rec3]

                        pd   = turn(pd)
                        gt   = turn(gt)

                        Prediction[i*TestBatchSize+k,:,:] = pd
                        GT[i*TestBatchSize+k,:,:] = gt
                        psnr = PSNR(pd,gt)
                        TotPSNR[0,total] = psnr
                        ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                        TotSSIM[0,total] = ssim

            PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            PlotPrediction(Prediction[num,:,:],vel_rec1, vel_rec2, vel_rec3,label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        if (scale.get()) == 5:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            for i in range(TestBatchSize):

                # Add noise to the original image
                images = images + noiseImg2
                images = torch.from_numpy(images)
                images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
                images = images.to(torch.float32)

                # Predictions
                net_noise.eval() 
                outputs  = net_noise(images,label_dsp_dim)
                outputs  = outputs.view(TestBatchSize,label_dsp_dim[0],label_dsp_dim[1])
                outputs  = outputs.data.cpu().numpy()
                gts      = labels.data.cpu().numpy()
                outputs1 = outputs

                # Calculate the PSNR, SSIM
                for k in range(TestBatchSize):
                    pd   = outputs[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                    gt   = gts[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])

                    vel_rec1 = float(var7.get())
                    vel_rec2 = float(var8.get())
                    vel_rec3 = float(var9.get())

                    vel_rec1 = (vel_rec1 * 100)
                    vel_rec2 = (vel_rec2 * 100)
                    vel_rec3 = (vel_rec3 * 100)
                    
                    frac1, whole1 = math.modf(vel_rec1)
                    frac2, whole2 = math.modf(vel_rec2)
                    frac3, whole3 = math.modf(vel_rec3)

                    if frac1 >= 0.5:
                        vel_rec1 = math.ceil(vel_rec1)
                    else:
                        vel_rec1 = round(vel_rec1)

                    if frac2 >= 0.5:
                        vel_rec2 = math.ceil(vel_rec2)
                    else:
                        vel_rec2 = round(vel_rec2)

                    if frac3 >= 0.5:
                        vel_rec3 = math.ceil(vel_rec3)
                    else:
                        vel_rec3 = round(vel_rec3)    

                    vel_rec1 = int(vel_rec1)
                    vel_rec2 = int(vel_rec2)
                    vel_rec3 = int(vel_rec3)

                    prof1 = outputs1[0,:,vel_rec1]
                    prof2 = outputs1[0,:,vel_rec2]
                    prof3 = outputs1[0,:,vel_rec3]

                    pd   = turn(pd)
                    gt   = turn(gt)

                    Prediction[i*TestBatchSize+k,:,:] = pd
                    GT[i*TestBatchSize+k,:,:] = gt
                    psnr = PSNR(pd,gt)
                    TotPSNR[0,total] = psnr
                    ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                    TotSSIM[0,total] = ssim

            PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            PlotPrediction(Prediction[num,:,:],vel_rec1, vel_rec2, vel_rec3,label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        elif (scale.get()) == 10:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            for i in range(TestBatchSize):

                # Add noise to the original image
                images = images + noiseImg3
                images = torch.from_numpy(images)
                images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
                images = images.to(torch.float32)

                # Predictions
                net_noise.eval() 
                outputs  = net_noise(images,label_dsp_dim)
                outputs  = outputs.view(TestBatchSize,label_dsp_dim[0],label_dsp_dim[1])
                outputs  = outputs.data.cpu().numpy()
                gts      = labels.data.cpu().numpy()
                outputs1 = outputs

                # Calculate the PSNR, SSIM
                for k in range(TestBatchSize):
                    pd   = outputs[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                    gt   = gts[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])

                    vel_rec1 = float(var7.get())
                    vel_rec2 = float(var8.get())
                    vel_rec3 = float(var9.get())

                    vel_rec1 = (vel_rec1 * 100)
                    vel_rec2 = (vel_rec2 * 100)
                    vel_rec3 = (vel_rec3 * 100)
                    
                    frac1, whole1 = math.modf(vel_rec1)
                    frac2, whole2 = math.modf(vel_rec2)
                    frac3, whole3 = math.modf(vel_rec3)

                    if frac1 >= 0.5:
                        vel_rec1 = math.ceil(vel_rec1)
                    else:
                        vel_rec1 = round(vel_rec1)

                    if frac2 >= 0.5:
                        vel_rec2 = math.ceil(vel_rec2)
                    else:
                        vel_rec2 = round(vel_rec2)

                    if frac3 >= 0.5:
                        vel_rec3 = math.ceil(vel_rec3)
                    else:
                        vel_rec3 = round(vel_rec3)    

                    vel_rec1 = int(vel_rec1)
                    vel_rec2 = int(vel_rec2)
                    vel_rec3 = int(vel_rec3)

                    prof1 = outputs1[0,:,vel_rec1]
                    prof2 = outputs1[0,:,vel_rec2]
                    prof3 = outputs1[0,:,vel_rec3]

                    pd   = turn(pd)
                    gt   = turn(gt)

                    Prediction[i*TestBatchSize+k,:,:] = pd
                    GT[i*TestBatchSize+k,:,:] = gt
                    psnr = PSNR(pd,gt)
                    TotPSNR[0,total] = psnr
                    ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                    TotSSIM[0,total] = ssim

            PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            PlotPrediction(Prediction[num,:,:],vel_rec1, vel_rec2, vel_rec3,label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        if (scale.get()) == 15:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            for i in range(TestBatchSize):

                # Add noise to the original image
                images = images + noiseImg4
                images = torch.from_numpy(images)
                images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
                images = images.to(torch.float32)

                # Predictions
                net_noise.eval() 
                outputs  = net_noise(images,label_dsp_dim)
                outputs  = outputs.view(TestBatchSize,label_dsp_dim[0],label_dsp_dim[1])
                outputs  = outputs.data.cpu().numpy()
                gts      = labels.data.cpu().numpy()
                outputs1 = outputs

                # Calculate the PSNR, SSIM
                for k in range(TestBatchSize):
                    pd   = outputs[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                    gt   = gts[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                
                    vel_rec1 = float(var7.get())
                    vel_rec2 = float(var8.get())
                    vel_rec3 = float(var9.get())

                    vel_rec1 = (vel_rec1 * 100)
                    vel_rec2 = (vel_rec2 * 100)
                    vel_rec3 = (vel_rec3 * 100)
                    
                    frac1, whole1 = math.modf(vel_rec1)
                    frac2, whole2 = math.modf(vel_rec2)
                    frac3, whole3 = math.modf(vel_rec3)

                    if frac1 >= 0.5:
                        vel_rec1 = math.ceil(vel_rec1)
                    else:
                        vel_rec1 = round(vel_rec1)

                    if frac2 >= 0.5:
                        vel_rec2 = math.ceil(vel_rec2)
                    else:
                        vel_rec2 = round(vel_rec2)

                    if frac3 >= 0.5:
                        vel_rec3 = math.ceil(vel_rec3)
                    else:
                        vel_rec3 = round(vel_rec3)    

                    vel_rec1 = int(vel_rec1)
                    vel_rec2 = int(vel_rec2)
                    vel_rec3 = int(vel_rec3)

                    prof1 = outputs1[0,:,vel_rec1]
                    prof2 = outputs1[0,:,vel_rec2]
                    prof3 = outputs1[0,:,vel_rec3]

                    pd   = turn(pd)
                    gt   = turn(gt)

                    Prediction[i*TestBatchSize+k,:,:] = pd
                    GT[i*TestBatchSize+k,:,:] = gt
                    psnr = PSNR(pd,gt)
                    TotPSNR[0,total] = psnr
                    ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                    TotSSIM[0,total] = ssim

            PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            PlotPrediction(Prediction[num,:,:],vel_rec1, vel_rec2, vel_rec3,label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        elif (scale.get()) == 20:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)
        
            for i in range(TestBatchSize):

                # Add noise to the original image
                images = images + noiseImg5
                images = torch.from_numpy(images)
                images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
                images = images.to(torch.float32)

                # Predictions
                net_noise.eval() 
                outputs  = net_noise(images,label_dsp_dim)
                outputs  = outputs.view(TestBatchSize,label_dsp_dim[0],label_dsp_dim[1])
                outputs  = outputs.data.cpu().numpy()
                gts      = labels.data.cpu().numpy()
                outputs1 = outputs

                # Calculate the PSNR, SSIM
                for k in range(TestBatchSize):
                    pd   = outputs[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                    gt   = gts[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])

                    vel_rec1 = float(var7.get())
                    vel_rec2 = float(var8.get())
                    vel_rec3 = float(var9.get())

                    vel_rec1 = (vel_rec1 * 100)
                    vel_rec2 = (vel_rec2 * 100)
                    vel_rec3 = (vel_rec3 * 100)
                    
                    frac1, whole1 = math.modf(vel_rec1)
                    frac2, whole2 = math.modf(vel_rec2)
                    frac3, whole3 = math.modf(vel_rec3)

                    if frac1 >= 0.5:
                        vel_rec1 = math.ceil(vel_rec1)
                    else:
                        vel_rec1 = round(vel_rec1)

                    if frac2 >= 0.5:
                        vel_rec2 = math.ceil(vel_rec2)
                    else:
                        vel_rec2 = round(vel_rec2)

                    if frac3 >= 0.5:
                        vel_rec3 = math.ceil(vel_rec3)
                    else:
                        vel_rec3 = round(vel_rec3)    

                    vel_rec1 = int(vel_rec1)
                    vel_rec2 = int(vel_rec2)
                    vel_rec3 = int(vel_rec3)

                    prof1 = outputs1[0,:,vel_rec1]
                    prof2 = outputs1[0,:,vel_rec2]
                    prof3 = outputs1[0,:,vel_rec3]

                    pd   = turn(pd)
                    gt   = turn(gt)

                    Prediction[i*TestBatchSize+k,:,:] = pd
                    GT[i*TestBatchSize+k,:,:] = gt
                    psnr = PSNR(pd,gt)
                    TotPSNR[0,total] = psnr
                    ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                    TotSSIM[0,total] = ssim

            PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            PlotPrediction(Prediction[num,:,:],vel_rec1, vel_rec2, vel_rec3,label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        if (scale.get()) == 25:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)

            for i in range(TestBatchSize):

                # Add noise to the original image
                images = images + noiseImg6
                images = torch.from_numpy(images)
                images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
                images = images.to(torch.float32)

                # Predictions
                net_noise.eval() 
                outputs  = net_noise(images,label_dsp_dim)
                outputs  = outputs.view(TestBatchSize,label_dsp_dim[0],label_dsp_dim[1])
                outputs  = outputs.data.cpu().numpy()
                gts      = labels.data.cpu().numpy()
                outputs1 = outputs

                # Calculate the PSNR, SSIM
                for k in range(TestBatchSize):
                    pd   = outputs[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                    gt   = gts[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                
                    vel_rec1 = float(var7.get())
                    vel_rec2 = float(var8.get())
                    vel_rec3 = float(var9.get())

                    vel_rec1 = (vel_rec1 * 100)
                    vel_rec2 = (vel_rec2 * 100)
                    vel_rec3 = (vel_rec3 * 100)
                    
                    frac1, whole1 = math.modf(vel_rec1)
                    frac2, whole2 = math.modf(vel_rec2)
                    frac3, whole3 = math.modf(vel_rec3)

                    if frac1 >= 0.5:
                        vel_rec1 = math.ceil(vel_rec1)
                    else:
                        vel_rec1 = round(vel_rec1)

                    if frac2 >= 0.5:
                        vel_rec2 = math.ceil(vel_rec2)
                    else:
                        vel_rec2 = round(vel_rec2)

                    if frac3 >= 0.5:
                        vel_rec3 = math.ceil(vel_rec3)
                    else:
                        vel_rec3 = round(vel_rec3)    

                    vel_rec1 = int(vel_rec1)
                    vel_rec2 = int(vel_rec2)
                    vel_rec3 = int(vel_rec3)

                    prof1 = outputs1[0,:,vel_rec1]
                    prof2 = outputs1[0,:,vel_rec2]
                    prof3 = outputs1[0,:,vel_rec3]

                    pd   = turn(pd)
                    gt   = turn(gt)

                    Prediction[i*TestBatchSize+k,:,:] = pd
                    GT[i*TestBatchSize+k,:,:] = gt
                    psnr = PSNR(pd,gt)
                    TotPSNR[0,total] = psnr
                    ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                    TotSSIM[0,total] = ssim

            PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            PlotPrediction(Prediction[num,:,:],vel_rec1, vel_rec2, vel_rec3,label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        if (scale.get()) == 30:

            # flatten images
            images = images.numpy()
            images = np.reshape(images, -1)


            for i in range(TestBatchSize):

                # Add noise to the original image
                images = images + noiseImg7
                images = torch.from_numpy(images)
                images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
                images = images.to(torch.float32)

                # Predictions
                net_noise.eval() 
                outputs  = net_noise(images,label_dsp_dim)
                outputs  = outputs.view(TestBatchSize,label_dsp_dim[0],label_dsp_dim[1])
                outputs  = outputs.data.cpu().numpy()
                gts      = labels.data.cpu().numpy()
                outputs1 = outputs

                # Calculate the PSNR, SSIM
                for k in range(TestBatchSize):
                    pd   = outputs[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])
                    gt   = gts[k,:,:].reshape(label_dsp_dim[0],label_dsp_dim[1])

                    vel_rec1 = float(var7.get())
                    vel_rec2 = float(var8.get())
                    vel_rec3 = float(var9.get())

                    vel_rec1 = (vel_rec1 * 100)
                    vel_rec2 = (vel_rec2 * 100)
                    vel_rec3 = (vel_rec3 * 100)
                    
                    frac1, whole1 = math.modf(vel_rec1)
                    frac2, whole2 = math.modf(vel_rec2)
                    frac3, whole3 = math.modf(vel_rec3)

                    if frac1 >= 0.5:
                        vel_rec1 = math.ceil(vel_rec1)
                    else:
                        vel_rec1 = round(vel_rec1)

                    if frac2 >= 0.5:
                        vel_rec2 = math.ceil(vel_rec2)
                    else:
                        vel_rec2 = round(vel_rec2)

                    if frac3 >= 0.5:
                        vel_rec3 = math.ceil(vel_rec3)
                    else:
                        vel_rec3 = round(vel_rec3)    

                    vel_rec1 = int(vel_rec1)
                    vel_rec2 = int(vel_rec2)
                    vel_rec3 = int(vel_rec3)

                    prof1 = outputs1[0,:,vel_rec1]
                    prof2 = outputs1[0,:,vel_rec2]
                    prof3 = outputs1[0,:,vel_rec3]

                    pd   = turn(pd)
                    gt   = turn(gt)

                    Prediction[i*TestBatchSize+k,:,:] = pd
                    GT[i*TestBatchSize+k,:,:] = gt
                    psnr = PSNR(pd,gt)
                    TotPSNR[0,total] = psnr
                    ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                    TotSSIM[0,total] = ssim

            PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            PlotPrediction(Prediction[num,:,:],vel_rec1, vel_rec2, vel_rec3,label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
            plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

        SaveTestResults(TotPSNR,TotSSIM,Prediction,GT,results_dir)

        # Record the consuming time
        time_elapsed = time.time() - since

    # screen.insert(INSERT, " " + '\n')
    # screen.insert(INSERT, "Noise-aware UNet Model Loaded ...." + '\n')
    # screen.insert(INSERT, " " + '\n')

    else:
        pass
    
    return psnr, ssim, time_elapsed

var3 = StringVar()
scale = tk.Scale(root, variable=var3, font=("helvetica", 11), orient='horizontal', bg = "gray80", bd=1, from_=0, to=30, 
                            tickinterval=5, resolution=5, length=200, command=())
scale.place(x=615, y=140)

screen = tkscrolled.ScrolledText(root, bd=2, wrap="word")
screen.place(x=487, y=770, width=953, height=220)
screen.configure(font=("courier", 12))

var4 = StringVar()
seis_receiver = Entry(root, textvariable=var4, font = ("Times", 12, "bold"), bg="gray80", justify=CENTER, highlightthickness=2)
seis_receiver.config(highlightbackground="red", highlightcolor="red")
seis_receiver.place(x=115, y=250, width=60, height=25)

var5 = StringVar()
seis_receiver = Entry(root, textvariable=var5, font = ("Times", 12, "bold"), bg="gray80", justify=CENTER, highlightthickness=2)
seis_receiver.config(highlightbackground="blue", highlightcolor="blue")
seis_receiver.place(x=207, y=250, width=60, height=25)

var6 = StringVar()
seis_receiver = Entry(root, textvariable=var6, font = ("Times", 12, "bold"), bg="gray80", justify=CENTER, highlightthickness=2)
seis_receiver.config(highlightbackground="black", highlightcolor="black")
seis_receiver.place(x=300, y=250, width=60, height=25)

seis_receiver_label = customtkinter.CTkLabel(master=root, text="Enter Seismic Receiver No. (Range: 0 - 300)", text_font=("helvetica", 13),
                            image=img3, compound=RIGHT, fg_color="gray80", corner_radius=1).place(x=45, y=190)

seis_set1 = customtkinter.CTkLabel(master=root, text="Set 1", text_font=("helvetica", 13), width=25, fg_color="gray80", corner_radius=1).place(x=122, y=220)
seis_set2 = customtkinter.CTkLabel(master=root, text="Set 2", text_font=("helvetica", 13), width=25, fg_color="gray80", corner_radius=1).place(x=214, y=220)
seis_set3 = customtkinter.CTkLabel(master=root, text="Set 3", text_font=("helvetica", 13), width=25, fg_color="gray80", corner_radius=1).place(x=307, y=220)

var7 = StringVar()
vel_model_receiver = Entry(root, textvariable=var7, font = ("Times", 12, "bold"), bg="gray80", justify=CENTER, highlightthickness=2)
vel_model_receiver.config(highlightbackground="red", highlightcolor="red")
vel_model_receiver.place(x=1083, y=250, width=60, height=25)

var8 = StringVar()
vel_model_receiver = Entry(root, textvariable=var8, font = ("Times", 12, "bold"), bg="gray80", justify=CENTER, highlightthickness=2)
vel_model_receiver.config(highlightbackground="blue", highlightcolor="blue")
vel_model_receiver.place(x=1175, y=250, width=60, height=25)

var9 = StringVar()
vel_model_receiver = Entry(root, textvariable=var9, font = ("Times", 12, "bold"), bg="gray80", justify=CENTER, highlightthickness=2)
vel_model_receiver.config(highlightbackground="black", highlightcolor="black")
vel_model_receiver.place(x=1268, y=250, width=60, height=25)

model_receiver_label = customtkinter.CTkLabel(master=root, text="Enter Velocity Distance No. (Range: 0 - 3) km", text_font=("helvetica", 13),
                            image=img3, compound=RIGHT, fg_color="gray80", corner_radius=1).place(x=1008, y=190)

vel_set1 = customtkinter.CTkLabel(master=root, text="Set 1", text_font=("helvetica", 13), width=25, fg_color="gray80", corner_radius=1).place(x=1090, y=220)
vel_set2 = customtkinter.CTkLabel(master=root, text="Set 2", text_font=("helvetica", 13), width=25, fg_color="gray80", corner_radius=1).place(x=1182, y=220)
vel_set3 = customtkinter.CTkLabel(master=root, text="Set 3", text_font=("helvetica", 13), width=25, fg_color="gray80", corner_radius=1).place(x=1275, y=220)

data_min = -20
data_max = 38
label_min = 1500
label_max = 4500

transform_valid_data = torchvision.transforms.Compose([
    T.LogTransform(k=1),
    T.MinMaxNormalize(T.log_transform(data_min, k=1), T.log_transform(data_max, k=1))
])

transform_valid_label = torchvision.transforms.Compose([
    T.MinMaxNormalize(label_min, label_max)
])

val_anno ='/home/pi/Desktop/exp/fcnvmb/UPFWI/relevant_files/salt_down_valid.txt'

# if val_anno[-3:] == 'txt':
#     dataset_valid = FWIDataset(
#     val_anno,
#     preload=True,
#     sample_ratio=1,
#     file_size=10,
#     transform_data=transform_valid_data,
#     transform_label=transform_valid_label
#     )
# else:
# dataset_valid = torch.load(val_anno)

# dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=1, shuffle=True)

# def orig_kimb_seis():

#     global noiseImg1, noiseImg2, noiseImg3, noiseImg4, noiseImg5, noiseImg6, noiseImg7
    
#     #set arguments
#     vis = True
#     vis_path = '/home/pi/Desktop/exp/fcnvmb/results/SaltResults'
#     vis_batch = 1
#     vis_sample = 0

#     data, label = iter(dataloader_valid).next()
#     data = data.type(torch.FloatTensor)
#     label = label.type(torch.FloatTensor)

#     data_np = data.numpy()

#     if (scale.get()) == 0:

#         # flatten images
#         images = np.reshape(data_np, -1)

#         # set the target SNR
#         target_SNR_dB = 0

#         # compute power of image values
#         image_power = images ** 2

#         # compute average image and convert to dB
#         avg_image_power = np.mean(image_power)
#         avg_image_dB = 10 * np.log10(avg_image_power)

#         # compute noise and convert to  dB
#         avg_noiseImg_dB = avg_image_dB - target_SNR_dB
#         avg_noiseImg_power = 10 ** (avg_noiseImg_dB / 10)

#         # generate sample of white noise
#         mean_noise = 0
#         noiseImg1 = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))

#         images = images + noiseImg1
#         images = images.reshape(10, 9, 1251, 101)

#         trace1 = images[1, 1, :, 30]
#         trace2 = images[1, 1, :, 60]
#         trace3 = images[1, 1, :, 90]

#         if vis:
#             batch_idx = 0

#             if vis and batch_idx < vis_batch:
#                 plot_kimb_seismic(images, f'{vis_path}/Seismic.png')
#                 plot_kimb_trace(trace1, trace2, trace3, f'{vis_path}/Trace.png')
       
#     if (scale.get()) == 5:

#         # flatten images
#         images = np.reshape(data_np, -1)

#         # set the target SNR
#         target_SNR_dB = 5

#         # compute power of image values
#         image_power = images ** 2

#         # compute average image and convert to dB
#         avg_image_power = np.mean(image_power)
#         avg_image_dB = 10 * np.log10(avg_image_power)

#         # compute noise and convert to  dB
#         avg_noiseImg_dB = avg_image_dB - target_SNR_dB
#         avg_noiseImg_power = 10 ** (avg_noiseImg_dB / 10)

#         # generate sample of white noise
#         mean_noise = 0
#         noiseImg2 = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))

#         images = images + noiseImg2
#         images = images.reshape(10, 9, 1251, 101)

#         trace1 = images[1, 1, :, 30]
#         trace2 = images[1, 1, :, 60]
#         trace3 = images[1, 1, :, 90]

#         if vis:
#             batch_idx = 0

#             if vis and batch_idx < vis_batch:
#                 plot_kimb_seismic(images, f'{vis_path}/Seismic.png')
#                 plot_kimb_trace(trace1, trace2, trace3, f'{vis_path}/Trace.png')

#     if (scale.get()) == 10:

#         # flatten images
#         images = np.reshape(data_np, -1)

#         # set the target SNR
#         target_SNR_dB = 10

#         # compute power of image values
#         image_power = images ** 2

#         # compute average image and convert to dB
#         avg_image_power = np.mean(image_power)
#         avg_image_dB = 10 * np.log10(avg_image_power)

#         # compute noise and convert to  dB
#         avg_noiseImg_dB = avg_image_dB - target_SNR_dB
#         avg_noiseImg_power = 10 ** (avg_noiseImg_dB / 10)

#         # generate sample of white noise
#         mean_noise = 0
#         noiseImg3 = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))

#         images = images + noiseImg3
#         images = images.reshape(10, 9, 1251, 101)

#         trace1 = images[1, 1, :, 30]
#         trace2 = images[1, 1, :, 60]
#         trace3 = images[1, 1, :, 90]

#         if vis:
#             batch_idx = 0

#             if vis and batch_idx < vis_batch:
#                 plot_kimb_seismic(images, f'{vis_path}/Seismic.png')
#                 plot_kimb_trace(trace1, trace2, trace3, f'{vis_path}/Trace.png')

#     if (scale.get()) == 15:

#         # flatten images
#         images = np.reshape(data_np, -1)

#         # set the target SNR
#         target_SNR_dB = 15

#         # compute power of image values
#         image_power = images ** 2

#         # compute average image and convert to dB
#         avg_image_power = np.mean(image_power)
#         avg_image_dB = 10 * np.log10(avg_image_power)

#         # compute noise and convert to  dB
#         avg_noiseImg_dB = avg_image_dB - target_SNR_dB
#         avg_noiseImg_power = 10 ** (avg_noiseImg_dB / 10)

#         # generate sample of white noise
#         mean_noise = 0
#         noiseImg4 = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))

#         images = images + noiseImg4
#         images = images.reshape(10, 9, 1251, 101)

#         trace1 = images[1, 1, :, 30]
#         trace2 = images[1, 1, :, 60]
#         trace3 = images[1, 1, :, 90]

#         if vis:
#             batch_idx = 0

#             if vis and batch_idx < vis_batch:
#                 plot_kimb_seismic(images, f'{vis_path}/Seismic.png')
#                 plot_kimb_trace(trace1, trace2, trace3, f'{vis_path}/Trace.png')

#     if (scale.get()) == 20:

#         # flatten images
#         images = np.reshape(data_np, -1)

#         # set the target SNR
#         target_SNR_dB = 20

#         # compute power of image values
#         image_power = images ** 2

#         # compute average image and convert to dB
#         avg_image_power = np.mean(image_power)
#         avg_image_dB = 10 * np.log10(avg_image_power)

#         # compute noise and convert to  dB
#         avg_noiseImg_dB = avg_image_dB - target_SNR_dB
#         avg_noiseImg_power = 10 ** (avg_noiseImg_dB / 10)

#         # generate sample of white noise
#         mean_noise = 0
#         noiseImg5 = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))

#         images = images + noiseImg5
#         images = images.reshape(10, 9, 1251, 101)

#         trace1 = images[1, 1, :, 30]
#         trace2 = images[1, 1, :, 60]
#         trace3 = images[1, 1, :, 90]

#         if vis:
#             batch_idx = 0

#             if vis and batch_idx < vis_batch:
#                 plot_kimb_seismic(images, f'{vis_path}/Seismic.png')
#                 plot_kimb_trace(trace1, trace2, trace3, f'{vis_path}/Trace.png')

#     if (scale.get()) == 25:

#         # flatten images
#         images = np.reshape(data_np, -1)

#         # set the target SNR
#         target_SNR_dB = 25

#         # compute power of image values
#         image_power = images ** 2

#         # compute average image and convert to dB
#         avg_image_power = np.mean(image_power)
#         avg_image_dB = 10 * np.log10(avg_image_power)

#         # compute noise and convert to  dB
#         avg_noiseImg_dB = avg_image_dB - target_SNR_dB
#         avg_noiseImg_power = 10 ** (avg_noiseImg_dB / 10)

#         # generate sample of white noise
#         mean_noise = 0
#         noiseImg6 = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))

#         images = images + noiseImg6
#         images = images.reshape(10, 9, 1251, 101)

#         trace1 = images[1, 1, :, 30]
#         trace2 = images[1, 1, :, 60]
#         trace3 = images[1, 1, :, 90]

#         if vis:
#             batch_idx = 0

#             if vis and batch_idx < vis_batch:
#                 plot_kimb_seismic(images, f'{vis_path}/Seismic.png')
#                 plot_kimb_trace(trace1, trace2, trace3, f'{vis_path}/Trace.png')

#     if (scale.get()) == 30:

#         # flatten images
#         images = np.reshape(data_np, -1)

#         # set the target SNR
#         target_SNR_dB = 30

#         # compute power of image values
#         image_power = images ** 2

#         # compute average image and convert to dB
#         avg_image_power = np.mean(image_power)
#         avg_image_dB = 10 * np.log10(avg_image_power)

#         # compute noise and convert to  dB
#         avg_noiseImg_dB = avg_image_dB - target_SNR_dB
#         avg_noiseImg_power = 10 ** (avg_noiseImg_dB / 10)

#         # generate sample of white noise
#         mean_noise = 0
#         noiseImg7 = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))

#         images = images + noiseImg7
#         images = images.reshape(10, 9, 1251, 101)

#         trace1 = images[1, 1, :, 30]
#         trace2 = images[1, 1, :, 60]
#         trace3 = images[1, 1, :, 90]

#         if vis:
#             batch_idx = 0

#             if vis and batch_idx < vis_batch:
#                 plot_kimb_seismic(images, f'{vis_path}/Seismic.png')
#                 plot_kimb_trace(trace1, trace2, trace3, f'{vis_path}/Trace.png')

def evaluate(model, dataloader, data_min, data_max, label_min, label_max,
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
            data = data.type(torch.FloatTensor)
            label = label.type(torch.FloatTensor)
            
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
                    plot_kimb_profile(prof1, prof2, prof3, f'{vis_path}/VelProfile.png')
            batch_idx += 1

    label, label_pred = np.concatenate(label_list), np.concatenate(label_pred_list)
    label_t, pred_t = torch.cat(label_tensor), torch.cat(pred_tensor)
    l1 = nn.L1Loss()
    MAE = l1(label_t, pred_t)
    print(f'MAE: {MAE}')
    l2 = nn.MSELoss()
    MSE = l2(label_t, pred_t)
    print(f'MSE: {MSE}')
    ssim_loss = pytorch_ssim.SSIM(window_size=11)
    SSIM = ssim_loss(label_t / 2 + 0.5, pred_t / 2 + 0.5)
    print(f'SSIM: {SSIM}') 

    if vis:
        vel, vel_pred = np.concatenate(vis_vel), np.concatenate(vis_pred)

    return  MAE, MSE, SSIM

def main():

    # set arguments
    output_path = '/home/pi/Desktop/exp/fcnvmb/results'
    os.makedirs(output_path, exist_ok=True)
    save_name = 'SaltResults'
    output_path = os.path.join(output_path, save_name)
    val_anno ='/home/pi/Desktop/exp/fcnvmb/UPFWI/relevant_files/salt_down_valid.txt'
    model = "FCN4_1"
    up_mode = None
    resume = '/home/pi/Desktop/exp/fcnvmb/UPFWI/checkpoints/fcn_kimb_1/run1/model_1000.pth'
    vis=True
    vis_suffix = '1000_test'
    vis_batch = 1
    vis_sample = 1

    utils.mkdir(output_path)

    # create validation dataset
    data_min = -20
    data_max = 38
    label_min = 1500
    label_max = 4500

    transform_valid_data = torchvision.transforms.Compose([
        T.LogTransform(k=1),
        T.MinMaxNormalize(T.log_transform(data_min, k=1), T.log_transform(data_max, k=1))
    ])

    transform_valid_label = torchvision.transforms.Compose([
        T.MinMaxNormalize(label_min, label_max)
    ])
    if val_anno[-3:] == 'txt':
        dataset_valid = FWIDataset(
            val_anno,
            preload=True,
            sample_ratio=1,
            file_size=10,
            transform_data=transform_valid_data,
            transform_label=transform_valid_label
        )
    else:
        dataset_valid = torch.load(val_anno)

    # create validation dataloader
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=1, shuffle=True)

    if model not in network.model_dict:
        print('Unsupported model.')
        sys.exit()

    if up_mode:    
        model = network.model_dict[model](upsample_mode=up_mode)
    else:
        model = network.model_dict[model]()

    if resume:
        checkpoint = torch.load(resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    
    if vis:
        vis_path = output_path
    
    start_time = time.time()
    if vis:
        MAE, MSE, SSIM = evaluate(model, dataloader_valid, data_min, data_max, label_min, label_max,
                                vis=True, vis_path=vis_path, vis_batch=vis_batch, vis_sample=vis_sample)
    else:
        MAE, MSE, SSIM = evaluate(model, dataloader_valid, data_min, data_max, label_min, label_max)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testing time {}'.format(total_time_str))
    
    return  MAE, MSE, SSIM, total_time_str

def screen_disp():
    if no_noise:
        psnr, ssim, time_elapsed = no_noise()
        psnr = "{:.2f}".format(psnr)
        ssim = "{:.4f}".format(ssim)
        time_elapsed = round(time_elapsed)

    # screen.insert(INSERT, "*******************************************" + '\n') 
    # screen.insert(INSERT, "            START TESTING                  " + '\n')
    # screen.insert(INSERT, "*******************************************" + '\n')  

    # time = f"The testing time is {time_elapsed} seconds"
    # test_results = f"The testing PSNR: {psnr}, SSIM: {ssim}"
    # screen.insert(END, time + '\n')
    # screen.insert(END, test_results + '\n')


def noise_screen_disp():
    if noise_aware:
        psnr, ssim, time_elapsed = noise_aware()
        psnr = "{:.2f}".format(psnr)
        ssim = "{:.4f}".format(ssim)
        time_elapsed = round(time_elapsed)

    # screen.insert(INSERT, "*******************************************" + '\n') 
    # screen.insert(INSERT, "            START TESTING                  " + '\n')
    # screen.insert(INSERT, "*******************************************" + '\n')  
    # time = f"The testing time is {time_elapsed} seconds"
    # test_results = f"The testing PSNR: {psnr}, SSIM: {ssim}"
    # screen.insert(END, time + '\n')
    # screen.insert(END, test_results + '\n')

def orig_screen_disp():
    if orig_no_noise_pred:
        psnr, ssim, time_elapsed = orig_no_noise_pred()
        psnr = "{:.2f}".format(psnr)
        ssim = "{:.4f}".format(ssim)
        time_elapsed = round(time_elapsed)

    screen.insert(INSERT, "*******************************************" + '\n') 
    screen.insert(INSERT, "            START TESTING                  " + '\n')
    screen.insert(INSERT, "*******************************************" + '\n')  

    time = f"The testing time is {time_elapsed} seconds"
    test_results = f"The testing PSNR: {psnr}, SSIM: {ssim}"
    screen.insert(END, time + '\n')
    screen.insert(END, test_results + '\n')

def orig_noise_screen_disp():
    if orig_noise_aware_pred:
        psnr, ssim, time_elapsed = orig_noise_aware_pred()
        psnr = "{:.2f}".format(psnr)
        ssim = "{:.4f}".format(ssim)
        time_elapsed = round(time_elapsed)

    screen.insert(INSERT, "*******************************************" + '\n') 
    screen.insert(INSERT, "            START TESTING                  " + '\n')
    screen.insert(INSERT, "*******************************************" + '\n')  
    time = f"The testing time is {time_elapsed} seconds"
    test_results = f"The testing PSNR: {psnr}, SSIM: {ssim}"
    screen.insert(END, time + '\n')
    screen.insert(END, test_results + '\n')

def orig_kimb_screen_disp():
    if main:
        mae, mse, ssim, time_elapsed = main()
        mae = "{:.4f}".format(mae)
        mse = "{:.4f}".format(mse)
        ssim = "{:.4f}".format(ssim)
        time_elapsed = round(time_elapsed)

    screen.insert(INSERT, "*******************************************" + '\n') 
    screen.insert(INSERT, "            START TESTING                  " + '\n')
    screen.insert(INSERT, "*******************************************" + '\n')  
    time = f"The testing time is {time_elapsed} seconds"
    test_results = f"The testing MAE: {mae}, MSE: {mse} and SSIM: {ssim}"
    screen.insert(END, time + '\n')
    screen.insert(END, test_results + '\n')

check_1 = IntVar()
check_2 = IntVar()
check_3 = IntVar() 

receive_button = customtkinter.CTkButton(master=root, image=img5, text="Receive Seismic Data", text_font=("helvetica", 13), width=220, height=25, 
                                    corner_radius=10, compound="right", fg_color="white", command=ReceiveCall)
receive_button.place(x=120, y=50)

pred_button = customtkinter.CTkButton(master=root, image=img6, text="Prediction", text_font=("helvetica", 13), width=220, height=25, compound="right", 
                                   corner_radius=10, fg_color="white", command=TestingPD)
pred_button.place(x=1095, y=50)

trace_button = customtkinter.CTkButton(master=root, image=img7, text="Plot Trace", text_font=("helvetica", 13), width=220, height=25,
                                    corner_radius=10, compound="right", fg_color="white", command=TraceCall)
trace_button.place(x=607, y=50)

vel_button = customtkinter.CTkButton(master=root, image=img8, text="Plot Velocity Profile", text_font=("helvetica", 13), width=220, height=25,
                                corner_radius=10, compound="right",fg_color = "white", command=VelocityCall)
vel_button.place(x=1573, y=50)

# noiseUNet_button = customtkinter.CTkButton(master=root, image=noise_aware_img, text="", corner_radius=1,
#                                 fg_color = "white", command=NoisePD)
# noiseUNet_button.place(x=650, y=870)

# annot_pred_button = customtkinter.CTkButton(master=root, image=img4, text="Prediction (No-noise)", width=220, height=25, compound="right", 
#                                    corner_radius=10, fg_color="white", command=AnnotTestingPD)
# annot_pred_button.place(x=125, y=870)

# annot_pred_noise_button = customtkinter.CTkButton(master=root, image=img11, text="Prediction (Noise-aware)", width=220, height=25, compound="right", 
#                                    corner_radius=10, fg_color="white", command=AnnotTestingPDNoise)
# annot_pred_noise_button.place(x=125, y=932)

# annot_seis_button = customtkinter.CTkButton(master=root, image=img10, text="Seismic Data", width=220, height=25, compound="right", 
#                                    corner_radius=10, fg_color="white", command=AnnotReceiverCall)
# annot_seis_button.place(x=125, y=808)
uncheck = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/exp/fcnvmb/images/uncheck.PNG'))
check = ImageTk.PhotoImage(Image.open('/home/pi/Desktop/exp/fcnvmb/images/check.PNG'))

annot_seis_check = Checkbutton(root, text = 'Annotate Seismic Data', font = ("helvetica", 13), bd=0, variable = check_1, compound="left", image=uncheck,
                selectimage=check, indicatoron=False, selectcolor="gray80", bg="gray80", activebackground="gray80", command=AnnotReceiverCall)
annot_seis_check.place(x=85, y=800)

annot_pred_check = Checkbutton(root, text = 'Annotate Prediction (No-noise model)', font = ("helvetica", 13), bd=0, compound="left", variable = check_2,
                image=uncheck, selectimage=check, indicatoron=False, selectcolor="gray80", bg="gray80", activebackground="gray80", command=AnnotTestingPD)
annot_pred_check.place(x=85, y=845)  

annot_pred_noise_check = Checkbutton(root, text = 'Annotate Prediction (Noise-aware model)', font = ("helvetica", 13), bd=0, variable = check_3, compound="left",
                    image=uncheck, selectimage=check, indicatoron=False, selectcolor="gray80", bg="gray80", activebackground="gray80", command=AnnotTestingPDNoise)
annot_pred_noise_check.place(x=85, y=890)

# model_button = customtkinter.CTkButton(master=root, image = Inv, text="", corner_radius=1, fg_color="white", 
#                                         command = combineFunc(lambda: evaluate(model, dataloader_valid, 
#                                         data_min, data_max, label_min, label_max),
#                                         lambda: args,
#                                         lambda: main(args)))
# model_button = customtkinter.CTkButton(master=root, image = Inv, text="", corner_radius=1, fg_color="white", 
#                                         command = main)                                       
# model_button.place(x=659, y=808)

# receive_kimb_button = customtkinter.CTkButton(master=root, text="Receive KImberlina Data", corner_radius=1, fg_color="white", 
#                                         command = orig_kimb_seis)                                       
# receive_kimb_button.place(x=645, y=932)

modelmenu = Menu(menubar, tearoff=0)
modelmenu.add_command(image=noise_aware_img, command=NoisePD)
modelmenu.add_command(image=inv, command=main)
menubar.add_cascade(label="Select Model", menu=modelmenu)

root.config(menu=menubar)
root.mainloop()
# -*- coding: utf-8 -*-
from email.mime import image
from tkinter import ttk
from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
import os
import customtkinter
import sys
import numpy as np
import subprocess
import FCNVMB_test
from ParamConfig import *
from PathConfig import *
from LibConfig import *

# root = Tk()
root = tk.Toplevel()
root.state("zoom")
blank_space = " "
root.title(238*blank_space+"Inversion GUI App")
canvas = tk.Canvas(root, height=2000, width=1900)
canvas.pack()

def restart():
    canvas1.delete(canv_img1)
    canvas2.delete(canv_img2)
    # pd_label.destroy()
    trace_label.destroy()
    vel_label.destroy()
    receive_button.config(state=NORMAL)
    pred_button.config(state=NORMAL)
    trace_button.config(state=NORMAL)
    vel_button.config(state=NORMAL)
    screen.delete('1.0', END)
  
menubar = Menu(root, font=("Times", "10", "bold"))
filemenu = Menu(menubar, tearoff=0)
newImg = ImageTk.PhotoImage(Image.open('C:\\Users\\DANIEL\\Desktop\\images\\new.png').resize((20, 20), Image.ANTIALIAS))
filemenu.add_command(label=" New", image=newImg, compound=LEFT, command=())
openImg = ImageTk.PhotoImage(Image.open('C:\\Users\\DANIEL\\Desktop\\images\\open.png').resize((20, 20), Image.ANTIALIAS))
filemenu.add_command(label=" Open...", image=openImg, compound=LEFT, command=())
saveImg = ImageTk.PhotoImage(Image.open('C:\\Users\\DANIEL\\Desktop\\images\\save.png').resize((20, 20), Image.ANTIALIAS))
filemenu.add_command(label=" Save", image=saveImg, compound=LEFT,  command=())
filemenu.add_separator()
resImg = ImageTk.PhotoImage(Image.open('C:\\Users\\DANIEL\\Desktop\\images\\restart.png').resize((20, 20), Image.ANTIALIAS))
filemenu.add_command(label=" Restart", image=resImg, compound=LEFT, command=restart)
exitImg = ImageTk.PhotoImage(Image.open('C:\\Users\\DANIEL\\Desktop\\images\\close.png').resize((20, 20), Image.ANTIALIAS))
filemenu.add_command(label=" Exit", image=exitImg, compound=LEFT, command=root.destroy)

menubar.add_cascade(label="File", menu=filemenu)

frame1 = customtkinter.CTkFrame(master=root, width=385, height=710, fg_color = "gray80", corner_radius=15, border_width=1, border_color="gray")
frame1.place(x=2, y=287)
frame1.pack_propagate(0)
seis = customtkinter.CTkLabel(master=frame1, text="Seismic Data", fg_color="gray70", corner_radius=8)
seis.pack(pady=5)
canvas1 = tk.Canvas(frame1, bg="gray80", bd=1, height=660, width=375, highlightbackground="gray80")
canvas1.pack()

frame2 = customtkinter.CTkFrame(master=root, width=385, height=710, fg_color = "gray80", corner_radius=15, border_width=1, border_color="gray")
frame2.place(x=776, y=287)
frame2.pack_propagate(0)
seis_trace = customtkinter.CTkLabel(master=frame2, text="Seismic Trace", fg_color="gray70", corner_radius=8)
seis_trace.pack(pady=5)

frame3 = customtkinter.CTkFrame(master=root, width=385, height=710, fg_color = "gray80", corner_radius=15, border_width=1, border_color="gray")
frame3.place(x=389, y=287)
frame3.pack_propagate(0)
pred_model = customtkinter.CTkLabel(master=frame3, text="Predicted Velocity Model", fg_color="gray70", corner_radius=8)
pred_model.pack(pady=5)
canvas2 = tk.Canvas(frame3, bg="gray80", bd=1, height=660, width=375, highlightbackground="gray80")
canvas2.pack()

frame4 = customtkinter.CTkFrame(master=root, width=385, height=710, fg_color = "gray80", corner_radius=15, border_width=1, border_color="gray")
frame4.place(x=1163, y=287)
frame4.pack_propagate(0)
vel_profile = customtkinter.CTkLabel(master=frame4, text="Velocity Profile", fg_color="gray70", corner_radius=8)
vel_profile.pack(pady=5)

frame5 = customtkinter.CTkFrame(master=root,  width=368, height=710, fg_color = "gray80", corner_radius=15, border_width=1, border_color="gray")
frame5.place(x=1550, y=287)
frame5.pack_propagate(0)
out_screen = customtkinter.CTkLabel(master=frame5, text="Output Display", fg_color="gray70", corner_radius=8)
out_screen.pack(pady=5)

seis_labelFrame = LabelFrame(root, text=" Seismic ", font=("helvetica", 10), labelanchor="n", height=280, width=385, bg="gray80", bd=4, relief=GROOVE)
seis_labelFrame.place(x=2, y=7)

pred_labelFrame = LabelFrame(root, text=" Velocity Model ", font=("helvetica", 10), labelanchor="n", height=280, width=385, bg="gray80", bd=4, relief=GROOVE)
pred_labelFrame.place(x=389, y=7)

trace_labelFrame = LabelFrame(root, text=" Trace ", font=("helvetica", 10), labelanchor="n", height=280, width=385, bg="gray80", bd=4, relief=GROOVE)
trace_labelFrame.place(x=776, y=7)

vel_labelFrame = LabelFrame(root, text=" Velocity Profile ", font=("helvetica", 10), labelanchor="n", height=280, width=385, bg="gray80", bd=4, relief=GROOVE)
vel_labelFrame.place(x=1163, y=7)

data_labelFrame = LabelFrame(root, text=" Select Data ", font=("helvetica", 10), labelanchor="n", height=280, width=184, bg="gray80", bd=4, relief=GROOVE)
data_labelFrame.place(x=1550, y=7)

model_labelFrame = LabelFrame(root, text=" Select Model ", font=("helvetica", 10), labelanchor="n", height=280, width=184, bg="gray80", bd=4, relief=GROOVE)
model_labelFrame.place(x=1735, y=7)

#Data Toggle selection
salt_on = True

label=customtkinter.CTkLabel(master=root, text="   Salt Loaded ...", fg_color="gray80", corner_radius=1)
label.place(x=1573, y=70)

# Define our switch function
def switch():
	global salt_on
	
	# Determine whether salt data is loaded or not
	if salt_on:
		salt_button.config(image = kim)
		label.config(text = "Kimberlina Loaded ...",
						fg_color = "gray80")
		salt_on = False
	else:
	
		salt_button.config(image = salt)
		label.config(text = "   Salt Loaded ...", fg_color = "gray80")
		salt_on = True

# Define Our Images
salt = PhotoImage(file = "C:\\Users\\DANIEL\\Desktop\\images\\salt.png")
kim = PhotoImage(file = "C:\\Users\\DANIEL\\Desktop\\images\\kim.png")

# Create A Button
salt_button = customtkinter.CTkButton(master=root, image = salt, text="", width=20, height=20, 
				corner_radius=10, fg_color="white", command = switch)
salt_button.place(x=1570, y=105)

# Model Toggle Selection
model = True

model_label=customtkinter.CTkLabel(master=root, text="    UNet Loaded ...", fg_color="gray80", corner_radius=1)
model_label.place(x=1750, y=70)

# Define our switch function
def model_switch():
	global model
	
	# Determine whether UNet model is loaded or not
	if model:
		model_button.config(image = Inv)
		model_label.config(text = "InversionNet Loaded ...",
						fg_color = "gray80")
		model = False
	else:
	
		model_button.config(image = unet)
		model_label.config(text = "    UNet Loaded ...", fg_color = "gray80")
		model = True

# Define Our Images
unet = PhotoImage(file = "C:\\Users\\DANIEL\\Desktop\\images\\unet.png")
Inv = PhotoImage(file = "C:\\Users\\DANIEL\\Desktop\\images\\inv.png")

# Create A Button
model_button = customtkinter.CTkButton(master=root, image = unet, text="", 
				corner_radius=1, fg_color="white", command = model_switch)
model_button.place(x=1763, y=105)

def PredImage():
    screen_disp()
    global pd, pd_label, PD, pred_button, canv_img2
    pred_button.config(state=DISABLED)
    PD = Image.open('C:\\Users\\DANIEL\\Desktop\\exp\\fcnvmb\\results\\SaltResults\\PD.png')
    pd = ImageTk.PhotoImage(PD.resize((380, 390), Image.ANTIALIAS))
#     # pd_label = Label(frame3, image=pd)
#     # pd_label.image = pd
#     # pd_label.place(x=750, y=0)
#     # pd_label.pack(side=TOP, padx=2, pady=80)
    canv_img2 = canvas2.create_image(193, 277, image=pd, anchor=CENTER)
    # return pd_label

def SeisImage():
    global seis, seis_label, Seis, receive_button, canv_img1
    receive_button.config(state=DISABLED)
    Seis = Image.open('C:\\Users\\DANIEL\\Desktop\\exp\\fcnvmb\\results\\SaltResults\\Seismic.png')
    seis = ImageTk.PhotoImage(Seis.resize((380, 390), Image.ANTIALIAS))
    # seis_label = Label(frame1, image=seis)
    # seis_label.image = seis
    # seis_label.place(x=100, y=0)
    # seis_label.pack(side=TOP, padx=2, pady=80)
    canv_img1 = canvas1.create_image(193, 277, image=seis, anchor=CENTER)
    # return seis_label

def TraceImage():
    global trac, trace_label, Trace, trace_button
    trace_button.config(state=DISABLED)
    Trace = Image.open('C:\\Users\\DANIEL\\Desktop\\exp\\fcnvmb\\results\\SaltResults\\Trace.png')
    trac = ImageTk.PhotoImage(Trace.resize((380, 390), Image.ANTIALIAS))
    trace_label = Label(frame2, image=trac)
    trace_label.image = trac
    trace_label.place(x=100, y=0)
    trace_label.pack(side=TOP, padx=2, pady=80)
    return trace_label

def ProfileImage():
    global vel, vel_label, Vel, vel_button
    vel_button.config(state=DISABLED)
    Vel = Image.open('C:\\Users\\DANIEL\\Desktop\\exp\\fcnvmb\\results\\SaltResults\\VelProfile.png')
    vel = ImageTk.PhotoImage(Vel.resize((380, 390), Image.ANTIALIAS))
    vel_label = Label(frame4, image=vel)
    vel_label.image = vel
    vel_label.place(x=100, y=0)
    vel_label.pack(side=TOP, padx=2, pady=80)
    return vel_label

def zoom_seis(zoom):
    global seisImg
    newsize = (Seis.size[0]* int(zoom), 
                Seis.size[1]*int(zoom))
    scaledseis = Seis.resize(newsize, Image.LINEAR)
    seisImg = ImageTk.PhotoImage(scaledseis)
    canvas1.itemconfig(canv_img1, image=seisImg)
    # seis_label.configure(image=seisImg, width=380, height=390)
    # seis_label.place(x=80, y=0)
    # seis_label.pack(side=TOP, padx=2, pady=80)

def zoom_pd(zoom):
    global pdImg
    newsize = (PD.size[0]* int(zoom), 
                PD.size[1]*int(zoom))
    scaledPD = PD.resize(newsize, Image.LINEAR)
    pdImg = ImageTk.PhotoImage(scaledPD)
    canvas2.itemconfig(canv_img2, image=pdImg)
    # pd_label.configure(image=pdImg, width=380, height=390)
    # pd_label.place(x=750, y=0)
    # pd_label.pack(side=TOP, padx=2, pady=80)

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
    SeisImage()

def TraceCall():
    TraceImage()

def VelocityCall():
    ProfileImage()

img1 = ImageTk.PhotoImage(Image.open('C:\\Users\\DANIEL\\Desktop\\images\\zoom1.png').resize((25, 25), Image.ANTIALIAS))
img2 = ImageTk.PhotoImage(Image.open('C:\\Users\\DANIEL\\Desktop\\images\\zoom2.png').resize((25, 25), Image.ANTIALIAS))
img3 = ImageTk.PhotoImage(Image.open('C:\\Users\\DANIEL\\Desktop\\images\\receiver.png').resize((25, 25), Image.ANTIALIAS))
img5 = ImageTk.PhotoImage(Image.open('C:\\Users\\DANIEL\\Desktop\\images\\data-receive.png').resize((25, 25), Image.ANTIALIAS))
img6 = ImageTk.PhotoImage(Image.open('C:\\Users\\DANIEL\\Desktop\\images\\prediction.png').resize((25, 25), Image.ANTIALIAS))
img7 = ImageTk.PhotoImage(Image.open('C:\\Users\\DANIEL\\Desktop\\images\\trace.JPG').resize((25, 25), Image.ANTIALIAS))
img8 = ImageTk.PhotoImage(Image.open('C:\\Users\\DANIEL\\Desktop\\images\\vel_prof.JPG').resize((25, 25), Image.ANTIALIAS))
img9 = ImageTk.PhotoImage(Image.open('C:\\Users\\DANIEL\\Desktop\\images\\noise.png').resize((100, 30), Image.ANTIALIAS))

seis_scaler_label = customtkinter.CTkLabel(master=root, text="Zoom Seismic ", width=220, image=img1, compound=RIGHT,  
                                        fg_color="gray80", corner_radius=1).place(x=85, y=110)

pd_scaler_label = customtkinter.CTkLabel(master=root, text="Zoom Prediction ", image=img2, compound=RIGHT,  
                                        fg_color="gray80", corner_radius=1).place(x=525, y=110)

var1 = StringVar()
seis_scale = tk.Scale(root, variable=var1, orient='horizontal', bg = "gray80", bd=1,
                        from_=1, to=5, length=200, resolution=1, command=zoom_seis)
seis_scale.place(x=90, y=140)

var2 = StringVar()
pd_scale = tk.Scale(root, variable=var2, orient='horizontal', bg = "gray80", bd=1, 
                        from_=1, to=5, length=200, resolution=1, command=zoom_pd)
pd_scale.place(x=485, y=140)

noise_scaler_label = customtkinter.CTkLabel(master=root, text="   Noise (dB)   ", width=100, image=img9, compound=RIGHT, 
                                    fg_color="gray80", corner_radius=1).place(x=860, y=105)

net = FCNVMB_test.net
test_set,label_set,data_dsp_dim,label_dsp_dim = DataLoad_Test(test_size=TestSize,test_data_dir=test_data_dir, \
                                                            data_dim=DataDim,in_channels=Inchannels, \
                                                            model_dim=ModelDim,data_dsp_blk=data_dsp_blk, \
                                                            label_dsp_blk=label_dsp_blk,start=121, \
                                                            datafilename=datafilename,dataname=dataname, \
                                                            truthfilename=truthfilename,truthname=truthname)
test        = data_utils.TensorDataset(torch.from_numpy(test_set),torch.from_numpy(label_set))
test_loader = data_utils.DataLoader(test,batch_size=1,shuffle=False)

def noise_scale(scale_value=0):
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
        noiseImg = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))

        for i in range(TestBatchSize):

            # Add noise to the original image
            images = images + noiseImg
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
                
                    seis = images.view(Inchannels, data_dsp_dim[0], data_dsp_dim[1])
                    seis1 = seis.numpy()
                    seis   = seis[1,:,:].numpy()

                    seis_receiver1 = int(var4.get())
                    seis_receiver2 = int(var5.get()) 
                    seis_receiver3 = int(var6.get())

                    vel_receiver1 = int(var7.get())
                    vel_receiver2 = int(var8.get())
                    vel_receiver3 = int(var9.get())

                    trace1 = seis1[1,:,seis_receiver1]
                    trace2 = seis1[1,:,seis_receiver2]
                    trace3 = seis1[1,:,seis_receiver3]

                    prof1 = outputs1[0,:,vel_receiver1]
                    prof2 = outputs1[0,:,vel_receiver2]
                    prof3 = outputs1[0,:,vel_receiver3]

                    pd   = turn(pd)
                    gt   = turn(gt)

                    Prediction[i*TestBatchSize+k,:,:] = pd
                    GT[i*TestBatchSize+k,:,:] = gt
                    psnr = PSNR(pd,gt)
                    TotPSNR[0,total] = psnr
                    ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                    TotSSIM[0,total] = ssim

        PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
        PlotPrediction(Prediction[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
        display_seismic(seis, data_dsp_dim, data_dsp_blk, dh, font2,font3,SavePath=results_dir)
        plot_trace(trace1, trace2, trace3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)
        plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

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
        noiseImg = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))

        for i in range(TestBatchSize):

            # Add noise to the original image
            images = images + noiseImg
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
            
                seis = images.view(Inchannels, data_dsp_dim[0], data_dsp_dim[1])
                seis1 = seis.numpy()
                seis   = seis[1,:,:].numpy()

                seis_receiver1 = int(var4.get())
                seis_receiver2 = int(var5.get()) 
                seis_receiver3 = int(var6.get())

                vel_receiver1 = int(var7.get())
                vel_receiver2 = int(var8.get())
                vel_receiver3 = int(var9.get())

                trace1 = seis1[1,:,seis_receiver1]
                trace2 = seis1[1,:,seis_receiver2]
                trace3 = seis1[1,:,seis_receiver3]

                prof1 = outputs1[0,:,vel_receiver1]
                prof2 = outputs1[0,:,vel_receiver2]
                prof3 = outputs1[0,:,vel_receiver3]

                pd   = turn(pd)
                gt   = turn(gt)

                Prediction[i*TestBatchSize+k,:,:] = pd
                GT[i*TestBatchSize+k,:,:] = gt
                psnr = PSNR(pd,gt)
                TotPSNR[0,total] = psnr
                ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                TotSSIM[0,total] = ssim

        PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
        PlotPrediction(Prediction[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
        display_seismic(seis, data_dsp_dim, data_dsp_blk, dh, font2,font3,SavePath=results_dir)
        plot_trace(trace1, trace2, trace3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)
        plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

    elif (scale.get()) == 10:

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
        noiseImg = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))

        for i in range(TestBatchSize):

            # Add noise to the original image
            images = images + noiseImg
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
            
                seis = images.view(Inchannels, data_dsp_dim[0], data_dsp_dim[1])
                seis1 = seis.numpy()
                seis   = seis[1,:,:].numpy()

                seis_receiver1 = int(var4.get())
                seis_receiver2 = int(var5.get()) 
                seis_receiver3 = int(var6.get())

                vel_receiver1 = int(var7.get())
                vel_receiver2 = int(var8.get())
                vel_receiver3 = int(var9.get())

                trace1 = seis1[1,:,seis_receiver1]
                trace2 = seis1[1,:,seis_receiver2]
                trace3 = seis1[1,:,seis_receiver3]

                prof1 = outputs1[0,:,vel_receiver1]
                prof2 = outputs1[0,:,vel_receiver2]
                prof3 = outputs1[0,:,vel_receiver3]

                pd   = turn(pd)
                gt   = turn(gt)

                Prediction[i*TestBatchSize+k,:,:] = pd
                GT[i*TestBatchSize+k,:,:] = gt
                psnr = PSNR(pd,gt)
                TotPSNR[0,total] = psnr
                ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                TotSSIM[0,total] = ssim

        PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
        PlotPrediction(Prediction[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
        display_seismic(seis, data_dsp_dim, data_dsp_blk, dh, font2,font3,SavePath=results_dir)
        plot_trace(trace1, trace2, trace3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)
        plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

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
        noiseImg = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))

        for i in range(TestBatchSize):

            # Add noise to the original image
            images = images + noiseImg
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
            
                seis = images.view(Inchannels, data_dsp_dim[0], data_dsp_dim[1])
                seis1 = seis.numpy()
                seis   = seis[1,:,:].numpy()

                seis_receiver1 = int(var4.get())
                seis_receiver2 = int(var5.get()) 
                seis_receiver3 = int(var6.get())

                vel_receiver1 = int(var7.get())
                vel_receiver2 = int(var8.get())
                vel_receiver3 = int(var9.get())

                trace1 = seis1[1,:,seis_receiver1]
                trace2 = seis1[1,:,seis_receiver2]
                trace3 = seis1[1,:,seis_receiver3]

                prof1 = outputs1[0,:,vel_receiver1]
                prof2 = outputs1[0,:,vel_receiver2]
                prof3 = outputs1[0,:,vel_receiver3]

                pd   = turn(pd)
                gt   = turn(gt)

                Prediction[i*TestBatchSize+k,:,:] = pd
                GT[i*TestBatchSize+k,:,:] = gt
                psnr = PSNR(pd,gt)
                TotPSNR[0,total] = psnr
                ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                TotSSIM[0,total] = ssim

        PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
        PlotPrediction(Prediction[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
        display_seismic(seis, data_dsp_dim, data_dsp_blk, dh, font2,font3,SavePath=results_dir)
        plot_trace(trace1, trace2, trace3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)
        plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

    elif (scale.get()) == 20:

        # flatten images
        images = images.numpy()
        images = np.reshape(images, -1)

        # set the target SNR
        target_SNR_dB = 20

        # compute power of image values
        image_power = images ** 2

        # compute average image power and convert to dB
        avg_image_power = np.mean(image_power)
        avg_image_dB = 10 * np.log10(avg_image_power)

        # compute noise and convert to  dB
        avg_noiseImg_dB = avg_image_dB - target_SNR_dB
        avg_noiseImg_power = 10 ** (avg_noiseImg_dB / 10)

        # generate sample of white noise
        mean_noise = 0
        noiseImg = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))
       
        for i in range(TestBatchSize):

            # Add noise to the original image
            images = images + noiseImg
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
            
                seis = images.view(Inchannels, data_dsp_dim[0], data_dsp_dim[1])
                seis1 = seis.numpy()
                seis   = seis[1,:,:].numpy()

                seis_receiver1 = int(var4.get())
                seis_receiver2 = int(var5.get()) 
                seis_receiver3 = int(var6.get())

                vel_receiver1 = int(var7.get())
                vel_receiver2 = int(var8.get())
                vel_receiver3 = int(var9.get())

                trace1 = seis1[1,:,seis_receiver1]
                trace2 = seis1[1,:,seis_receiver2]
                trace3 = seis1[1,:,seis_receiver3]

                prof1 = outputs1[0,:,vel_receiver1]
                prof2 = outputs1[0,:,vel_receiver2]
                prof3 = outputs1[0,:,vel_receiver3]

                pd   = turn(pd)
                gt   = turn(gt)

                Prediction[i*TestBatchSize+k,:,:] = pd
                GT[i*TestBatchSize+k,:,:] = gt
                psnr = PSNR(pd,gt)
                TotPSNR[0,total] = psnr
                ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                TotSSIM[0,total] = ssim

        PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
        PlotPrediction(Prediction[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
        display_seismic(seis, data_dsp_dim, data_dsp_blk, dh, font2,font3,SavePath=results_dir)
        plot_trace(trace1, trace2, trace3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)
        plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

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
        noiseImg = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))

        for i in range(TestBatchSize):

            # Add noise to the original image
            images = images + noiseImg
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
            
                seis = images.view(Inchannels, data_dsp_dim[0], data_dsp_dim[1])
                seis1 = seis.numpy()
                seis   = seis[1,:,:].numpy()

                seis_receiver1 = int(var4.get())
                seis_receiver2 = int(var5.get()) 
                seis_receiver3 = int(var6.get())

                vel_receiver1 = int(var7.get())
                vel_receiver2 = int(var8.get())
                vel_receiver3 = int(var9.get())

                trace1 = seis1[1,:,seis_receiver1]
                trace2 = seis1[1,:,seis_receiver2]
                trace3 = seis1[1,:,seis_receiver3]

                prof1 = outputs1[0,:,vel_receiver1]
                prof2 = outputs1[0,:,vel_receiver2]
                prof3 = outputs1[0,:,vel_receiver3]

                pd   = turn(pd)
                gt   = turn(gt)

                Prediction[i*TestBatchSize+k,:,:] = pd
                GT[i*TestBatchSize+k,:,:] = gt
                psnr = PSNR(pd,gt)
                TotPSNR[0,total] = psnr
                ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                TotSSIM[0,total] = ssim

        PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
        PlotPrediction(Prediction[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
        display_seismic(seis, data_dsp_dim, data_dsp_blk, dh, font2,font3,SavePath=results_dir)
        plot_trace(trace1, trace2, trace3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)
        plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

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
        noiseImg = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))

        for i in range(TestBatchSize):

            # Add noise to the original image
            images = images + noiseImg
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
            
                seis = images.view(Inchannels, data_dsp_dim[0], data_dsp_dim[1])
                seis1 = seis.numpy()
                seis   = seis[1,:,:].numpy()

                seis_receiver1 = int(var4.get())
                seis_receiver2 = int(var5.get()) 
                seis_receiver3 = int(var6.get())

                vel_receiver1 = int(var7.get())
                vel_receiver2 = int(var8.get())
                vel_receiver3 = int(var9.get())

                trace1 = seis1[1,:,seis_receiver1]
                trace2 = seis1[1,:,seis_receiver2]
                trace3 = seis1[1,:,seis_receiver3]

                prof1 = outputs1[0,:,vel_receiver1]
                prof2 = outputs1[0,:,vel_receiver2]
                prof3 = outputs1[0,:,vel_receiver3]

                pd   = turn(pd)
                gt   = turn(gt)

                Prediction[i*TestBatchSize+k,:,:] = pd
                GT[i*TestBatchSize+k,:,:] = gt
                psnr = PSNR(pd,gt)
                TotPSNR[0,total] = psnr
                ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                TotSSIM[0,total] = ssim

        PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
        PlotPrediction(Prediction[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
        display_seismic(seis, data_dsp_dim, data_dsp_blk, dh, font2,font3,SavePath=results_dir)
        plot_trace(trace1, trace2, trace3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)
        plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

    SaveTestResults(TotPSNR,TotSSIM,Prediction,GT,results_dir)

    # Record the consuming time
    time_elapsed = time.time() - since

    return psnr, ssim, time_elapsed

def noise_aware(scale_value=0):

    model_file     = noise_models_dir+modelname+'_epoch'+str(Epochs)+'.pkl'
    net            = UnetModel(n_classes=Nclasses,in_channels=Inchannels, \
                           is_deconv=True,is_batchnorm=True)
    net.load_state_dict(torch.load(model_file,map_location=torch.device('cpu')))

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
        noiseImg = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))

        for i in range(TestBatchSize):

            # Add noise to the original image
            images = images + noiseImg
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
            
                seis = images.view(Inchannels, data_dsp_dim[0], data_dsp_dim[1])
                seis1 = seis.numpy()
                seis   = seis[1,:,:].numpy()

                seis_receiver1 = int(var4.get())
                seis_receiver2 = int(var5.get()) 
                seis_receiver3 = int(var6.get())

                vel_receiver1 = int(var7.get())
                vel_receiver2 = int(var8.get())
                vel_receiver3 = int(var9.get())

                trace1 = seis1[1,:,seis_receiver1]
                trace2 = seis1[1,:,seis_receiver2]
                trace3 = seis1[1,:,seis_receiver3]

                prof1 = outputs1[0,:,vel_receiver1]
                prof2 = outputs1[0,:,vel_receiver2]
                prof3 = outputs1[0,:,vel_receiver3]

                pd   = turn(pd)
                gt   = turn(gt)

                Prediction[i*TestBatchSize+k,:,:] = pd
                GT[i*TestBatchSize+k,:,:] = gt
                psnr = PSNR(pd,gt)
                TotPSNR[0,total] = psnr
                ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                TotSSIM[0,total] = ssim

        PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
        PlotPrediction(Prediction[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
        display_seismic(seis, data_dsp_dim, data_dsp_blk, dh, font2,font3,SavePath=results_dir)
        plot_trace(trace1, trace2, trace3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)
        plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

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
        noiseImg = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))

        for i in range(TestBatchSize):

            # Add noise to the original image
            images = images + noiseImg
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
            
                seis = images.view(Inchannels, data_dsp_dim[0], data_dsp_dim[1])
                seis1 = seis.numpy()
                seis   = seis[1,:,:].numpy()

                seis_receiver1 = int(var4.get())
                seis_receiver2 = int(var5.get()) 
                seis_receiver3 = int(var6.get())

                vel_receiver1 = int(var7.get())
                vel_receiver2 = int(var8.get())
                vel_receiver3 = int(var9.get())

                trace1 = seis1[1,:,seis_receiver1]
                trace2 = seis1[1,:,seis_receiver2]
                trace3 = seis1[1,:,seis_receiver3]

                prof1 = outputs1[0,:,vel_receiver1]
                prof2 = outputs1[0,:,vel_receiver2]
                prof3 = outputs1[0,:,vel_receiver3]

                pd   = turn(pd)
                gt   = turn(gt)

                Prediction[i*TestBatchSize+k,:,:] = pd
                GT[i*TestBatchSize+k,:,:] = gt
                psnr = PSNR(pd,gt)
                TotPSNR[0,total] = psnr
                ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                TotSSIM[0,total] = ssim

        PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
        PlotPrediction(Prediction[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
        display_seismic(seis, data_dsp_dim, data_dsp_blk, dh, font2,font3,SavePath=results_dir)
        plot_trace(trace1, trace2, trace3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)
        plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)
        
    elif (scale.get()) == 10:

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
        noiseImg = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))

        for i in range(TestBatchSize):

            # Add noise to the original image
            images = images + noiseImg
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
            
                seis = images.view(Inchannels, data_dsp_dim[0], data_dsp_dim[1])
                seis1 = seis.numpy()
                seis   = seis[1,:,:].numpy()

                seis_receiver1 = int(var4.get())
                seis_receiver2 = int(var5.get()) 
                seis_receiver3 = int(var6.get())

                vel_receiver1 = int(var7.get())
                vel_receiver2 = int(var8.get())
                vel_receiver3 = int(var9.get())

                trace1 = seis1[1,:,seis_receiver1]
                trace2 = seis1[1,:,seis_receiver2]
                trace3 = seis1[1,:,seis_receiver3]

                prof1 = outputs1[0,:,vel_receiver1]
                prof2 = outputs1[0,:,vel_receiver2]
                prof3 = outputs1[0,:,vel_receiver3]

                pd   = turn(pd)
                gt   = turn(gt)

                Prediction[i*TestBatchSize+k,:,:] = pd
                GT[i*TestBatchSize+k,:,:] = gt
                psnr = PSNR(pd,gt)
                TotPSNR[0,total] = psnr
                ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                TotSSIM[0,total] = ssim

        PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
        PlotPrediction(Prediction[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
        display_seismic(seis, data_dsp_dim, data_dsp_blk, dh, font2,font3,SavePath=results_dir)
        plot_trace(trace1, trace2, trace3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)
        plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)
        
    elif (scale.get()) == 15:

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
        noiseImg = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))

        for i in range(TestBatchSize):

            # Add noise to the original image
            images = images + noiseImg
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
            
                seis = images.view(Inchannels, data_dsp_dim[0], data_dsp_dim[1])
                seis1 = seis.numpy()
                seis   = seis[1,:,:].numpy()

                seis_receiver1 = int(var4.get())
                seis_receiver2 = int(var5.get()) 
                seis_receiver3 = int(var6.get())

                vel_receiver1 = int(var7.get())
                vel_receiver2 = int(var8.get())
                vel_receiver3 = int(var9.get())

                trace1 = seis1[1,:,seis_receiver1]
                trace2 = seis1[1,:,seis_receiver2]
                trace3 = seis1[1,:,seis_receiver3]

                prof1 = outputs1[0,:,vel_receiver1]
                prof2 = outputs1[0,:,vel_receiver2]
                prof3 = outputs1[0,:,vel_receiver3]

                pd   = turn(pd)
                gt   = turn(gt)

                Prediction[i*TestBatchSize+k,:,:] = pd
                GT[i*TestBatchSize+k,:,:] = gt
                psnr = PSNR(pd,gt)
                TotPSNR[0,total] = psnr
                ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                TotSSIM[0,total] = ssim

        PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
        PlotPrediction(Prediction[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
        display_seismic(seis, data_dsp_dim, data_dsp_blk, dh, font2,font3,SavePath=results_dir)
        plot_trace(trace1, trace2, trace3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)
        plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

    elif (scale.get()) == 20:

        # flatten images
        images = images.numpy()
        images = np.reshape(images, -1)

        # set the target SNR
        target_SNR_dB = 20

        # compute power of image values
        image_power = images ** 2

        # compute average image power and convert to dB
        avg_image_power = np.mean(image_power)
        avg_image_dB = 10 * np.log10(avg_image_power)

        # compute noise and convert to  dB
        avg_noiseImg_dB = avg_image_dB - target_SNR_dB
        avg_noiseImg_power = 10 ** (avg_noiseImg_dB / 10)

        # generate sample of white noise
        mean_noise = 0
        noiseImg = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))
       
        for i in range(TestBatchSize):

            # Add noise to the original image
            images = images + noiseImg
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
            
                seis = images.view(Inchannels, data_dsp_dim[0], data_dsp_dim[1])
                seis1 = seis.numpy()
                seis   = seis[1,:,:].numpy()

                seis_receiver1 = int(var4.get())
                seis_receiver2 = int(var5.get()) 
                seis_receiver3 = int(var6.get())

                vel_receiver1 = int(var7.get())
                vel_receiver2 = int(var8.get())
                vel_receiver3 = int(var9.get())

                trace1 = seis1[1,:,seis_receiver1]
                trace2 = seis1[1,:,seis_receiver2]
                trace3 = seis1[1,:,seis_receiver3]

                prof1 = outputs1[0,:,vel_receiver1]
                prof2 = outputs1[0,:,vel_receiver2]
                prof3 = outputs1[0,:,vel_receiver3]

                pd   = turn(pd)
                gt   = turn(gt)

                Prediction[i*TestBatchSize+k,:,:] = pd
                GT[i*TestBatchSize+k,:,:] = gt
                psnr = PSNR(pd,gt)
                TotPSNR[0,total] = psnr
                ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                TotSSIM[0,total] = ssim
               
        PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
        PlotPrediction(Prediction[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
        display_seismic(seis, data_dsp_dim, data_dsp_blk, dh, font2,font3,SavePath=results_dir)
        plot_trace(trace1, trace2, trace3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)
        plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

    elif (scale.get()) == 25:

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
        noiseImg = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))

        for i in range(TestBatchSize):

            # Add noise to the original image
            images = images + noiseImg
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
            
                seis = images.view(Inchannels, data_dsp_dim[0], data_dsp_dim[1])
                seis1 = seis.numpy()
                seis   = seis[1,:,:].numpy()

                seis_receiver1 = int(var4.get())
                seis_receiver2 = int(var5.get()) 
                seis_receiver3 = int(var6.get())

                vel_receiver1 = int(var7.get())
                vel_receiver2 = int(var8.get())
                vel_receiver3 = int(var9.get())

                trace1 = seis1[1,:,seis_receiver1]
                trace2 = seis1[1,:,seis_receiver2]
                trace3 = seis1[1,:,seis_receiver3]

                prof1 = outputs1[0,:,vel_receiver1]
                prof2 = outputs1[0,:,vel_receiver2]
                prof3 = outputs1[0,:,vel_receiver3]

                pd   = turn(pd)
                gt   = turn(gt)

                Prediction[i*TestBatchSize+k,:,:] = pd
                GT[i*TestBatchSize+k,:,:] = gt
                psnr = PSNR(pd,gt)
                TotPSNR[0,total] = psnr
                ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                TotSSIM[0,total] = ssim

        PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
        PlotPrediction(Prediction[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
        display_seismic(seis, data_dsp_dim, data_dsp_blk, dh, font2,font3,SavePath=results_dir)
        plot_trace(trace1, trace2, trace3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)
        plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

    elif (scale.get()) == 30:

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
        noiseImg = np.random.normal(mean_noise, np.sqrt(avg_noiseImg_power), len(image_power))

        for i in range(TestBatchSize):

            # Add noise to the original image
            images = images + noiseImg
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
            
                seis = images.view(Inchannels, data_dsp_dim[0], data_dsp_dim[1])
                seis1 = seis.numpy()
                seis   = seis[1,:,:].numpy()

                seis_receiver1 = int(var4.get())
                seis_receiver2 = int(var5.get()) 
                seis_receiver3 = int(var6.get())

                vel_receiver1 = int(var7.get())
                vel_receiver2 = int(var8.get())
                vel_receiver3 = int(var9.get())

                trace1 = seis1[1,:,seis_receiver1]
                trace2 = seis1[1,:,seis_receiver2]
                trace3 = seis1[1,:,seis_receiver3]

                prof1 = outputs1[0,:,vel_receiver1]
                prof2 = outputs1[0,:,vel_receiver2]
                prof3 = outputs1[0,:,vel_receiver3]

                pd   = turn(pd)
                gt   = turn(gt)

                Prediction[i*TestBatchSize+k,:,:] = pd
                GT[i*TestBatchSize+k,:,:] = gt
                psnr = PSNR(pd,gt)
                TotPSNR[0,total] = psnr
                ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
                TotSSIM[0,total] = ssim

        PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
        PlotPrediction(Prediction[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
        display_seismic(seis, data_dsp_dim, data_dsp_blk, dh, font2,font3,SavePath=results_dir)
        plot_trace(trace1, trace2, trace3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)
        plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)

    SaveTestResults(TotPSNR,TotSSIM,Prediction,GT,results_dir)

    # Record the consuming time
    time_elapsed = time.time() - since

    PredImage()

    return psnr, ssim, time_elapsed

var3 = StringVar()
scale = tk.Scale(root, variable=var3, orient='horizontal', bg = "gray80", bd=1, repeatdelay=1000000000,
                            from_=0, to=30, tickinterval=5, resolution=5, length=200, command=noise_scale)
scale.place(x=862, y=140)

screen = Text(root, bd=2)
screen.place(x=1552, y=325, width=363, height=660)

var4 = StringVar()
seis_receiver = Entry(root, bd=4, textvariable=var4, font = ("Times", 12, "bold"), bg="gray80", justify=CENTER, relief=GROOVE, highlightbackground="white")
seis_receiver.place(x=68, y=250, width=60, height=25)

var5 = StringVar()
seis_receiver = Entry(root, bd=4, textvariable=var5, font = ("Times", 12, "bold"), bg="gray80", justify=CENTER, relief=GROOVE, highlightbackground="white")
seis_receiver.place(x=160, y=250, width=60, height=25)

var6 = StringVar()
seis_receiver = Entry(root, bd=4, textvariable=var6, font = ("Times", 12, "bold"), bg="gray80", justify=CENTER, relief=GROOVE, highlightbackground="white")
seis_receiver.place(x=253, y=250, width=60, height=25)

seis_receiver_label = customtkinter.CTkLabel(master=root, text="Enter Seismic Receiver No. (Range: 0 - 300)", 
                            image=img3, compound=RIGHT, fg_color="gray80", corner_radius=1).place(x=55, y=200)

seis_set1 = customtkinter.CTkLabel(master=root, text="Set 1", width=25, fg_color="gray80", corner_radius=1).place(x=85, y=225)
seis_set2 = customtkinter.CTkLabel(master=root, text="Set 2", width=25, fg_color="gray80", corner_radius=1).place(x=177, y=225)
seis_set3 = customtkinter.CTkLabel(master=root, text="Set 3", width=25, fg_color="gray80", corner_radius=1).place(x=270, y=225)

var7 = StringVar()
vel_model_receiver = Entry(root, bd=4, textvariable=var7, font = ("Times", 12, "bold"), bg="gray80", justify=CENTER, relief=GROOVE, highlightbackground="white")
vel_model_receiver.place(x=453, y=250, width=60, height=25)

var8 = StringVar()
vel_model_receiver = Entry(root, bd=4, textvariable=var8, font = ("Times", 12, "bold"), bg="gray80", justify=CENTER, relief=GROOVE, highlightbackground="white")
vel_model_receiver.place(x=545, y=250, width=60, height=25)

var9 = StringVar()
vel_model_receiver = Entry(root, bd=4, textvariable=var9, font = ("Times", 12, "bold"), bg="gray80", justify=CENTER, relief=GROOVE, highlightbackground="white")
vel_model_receiver.place(x=638, y=250, width=60, height=25)

model_receiver_label = customtkinter.CTkLabel(master=root, text="Enter Velocity Receiver No. (Range: 0 - 300)",
                            image=img3, compound=RIGHT, fg_color="gray80", corner_radius=1).place(x=440, y=200)

vel_set1 = customtkinter.CTkLabel(master=root, text="Set 1", width=25, fg_color="gray80", corner_radius=1).place(x=470, y=225)
vel_set2 = customtkinter.CTkLabel(master=root, text="Set 2", width=25, fg_color="gray80", corner_radius=1).place(x=562, y=225)
vel_set3 = customtkinter.CTkLabel(master=root, text="Set 3", width=25, fg_color="gray80", corner_radius=1).place(x=655, y=225)

def screen_disp():
    if noise_scale:
        psnr, ssim, time_elapsed = noise_scale()
        psnr = "{:.2f}".format(psnr)
        ssim = "{:.4f}".format(ssim)
        time_elapsed = round(time_elapsed)

    elif noise_aware:
        psnr, ssim, time_elapsed = noise_aware()
        psnr = "{:.2f}".format(psnr)
        ssim = "{:.4f}".format(ssim)
        time_elapsed = round(time_elapsed)
        screen.insert(INSERT, "Noise-aware UNet Model Loaded ...." + '\n')

    time = f"The testing time is {time_elapsed} seconds"
    test_results = f"The testing PSNR: {psnr}, SSIM: {ssim}"
    screen.insert(END, time + '\n')
    screen.insert(END, test_results + '\n')

receive_button = customtkinter.CTkButton(master=root, image=img5, text="Receive Seismic Data", width=220, height=25, 
                                    corner_radius=10, compound="right", fg_color="white", command=ReceiveCall)
receive_button.place(x=85, y=50)

pred_button = customtkinter.CTkButton(master=root, image=img6, text="Prediction", width=220, height=25, compound="right", 
                                   corner_radius=10, fg_color="white", command=TestingPD)
pred_button.place(x=478, y=50)

trace_button = customtkinter.CTkButton(master=root, image=img7, text="Plot Trace",width=220, height=25,
                                    corner_radius=10, compound="right", fg_color="white", command=TraceCall)
trace_button.place(x=855, y=50)

vel_button = customtkinter.CTkButton(master=root, image=img8, text="Plot Velocity Profile", width=220, height=25,
                                corner_radius=10, compound="right",fg_color = "white", command=VelocityCall)
vel_button.place(x=1245, y=50)

noise_aware_img = PhotoImage(file = "C:\\Users\\DANIEL\\Desktop\\images\\noise_aware.PNG")

noiseUNet_button = customtkinter.CTkButton(master=root, image=noise_aware_img, text="", corner_radius=1,
                                fg_color = "white", command=noise_aware)
noiseUNet_button.place(x=1755, y=190)

root.config(menu=menubar)
root.mainloop()
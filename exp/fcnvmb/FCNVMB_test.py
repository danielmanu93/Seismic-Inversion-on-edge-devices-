# -*- coding: utf-8 -*-
"""
Fully Convolutional neural network (U-Net) for velocity model building from prestack

unmigrated seismic data directly


"""
################################################
########        IMPORT LIBARIES         ########
################################################
from ParamConfig import *
from PathConfig import *
from LibConfig import *
import matplotlib.pyplot as plt
################################################
########         LOAD    NETWORK        ########
################################################

# Here indicating the GPU you want to use. if you don't have GPU, just leave it.
cuda_available = torch.cuda.is_available()
device         = torch.device("cuda" if cuda_available else "cpu")
model_file     = no_noise_models_dir+modelname+'_epoch'+str(Epochs)+'.pkl'
net            = UnetModel(n_classes=Nclasses,in_channels=Inchannels, \
                           is_deconv=True,is_batchnorm=True) 
net.load_state_dict(torch.load(model_file,map_location=torch.device('cpu')))
#net.load_state_dict(torch.load(model_file))
if torch.cuda.is_available():
    net.cuda()

################################################
########    LOADING TESTING DATA       ########
################################################
print('***************** Loading Testing DataSet *****************')

test_set,label_set,data_dsp_dim,label_dsp_dim = DataLoad_Test(test_size=TestSize,test_data_dir=test_data_dir, \
                                                              data_dim=DataDim,in_channels=Inchannels, \
                                                              model_dim=ModelDim,data_dsp_blk=data_dsp_blk, \
                                                              label_dsp_blk=label_dsp_blk,start=121, \
                                                              datafilename=datafilename,dataname=dataname, \
                                                              truthfilename=truthfilename,truthname=truthname)

test        = data_utils.TensorDataset(torch.from_numpy(test_set),torch.from_numpy(label_set))
test_loader = data_utils.DataLoader(test,batch_size=TestBatchSize,shuffle=True)
# data = np.load('C:\\Users\\DANIEL\\Desktop\\exp\\fcnvmb\\data\\test_data\\SaltData\\georec_test\\data121.npy')
# data = data[1,:,:]
# print(data.shape)
# # # plt.imshow(data[5,:,:]) # seismic 
# plt.plot(data[1,:,110]) #trace
# plt.show()
################################################
########            TESTING             ########
################################################

print() 
print('*******************************************') 
print('*******************************************') 
print('            START TESTING                  ') 
print('*******************************************') 
print('*******************************************') 
print()

# Initialization
since      = time.time()
TotPSNR    = np.zeros((1,TestSize),dtype=float) 
TotSSIM    = np.zeros((1,TestSize),dtype=float) 
Prediction = np.zeros((TestSize,label_dsp_dim[0],label_dsp_dim[1]),dtype=float)
GT         = np.zeros((TestSize,label_dsp_dim[0],label_dsp_dim[1]),dtype=float)
Seismic    = np.zeros((TestSize,data_dsp_dim[0],data_dsp_dim[1]),dtype=float) 
total      = 0

# for i, (images,labels) in enumerate(test_loader):   
for i in range(TestBatchSize): 

    images, labels = iter(test_loader).next()
    images = images.view(TestBatchSize,Inchannels,data_dsp_dim[0],data_dsp_dim[1])
    labels = labels.view(TestBatchSize,Nclasses,label_dsp_dim[0],label_dsp_dim[1])
    images = images.to(device)
    labels = labels.to(device)
    
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
        # print(seis1)
        seis   = seis[1,:,:].numpy()

        trace1 = seis1[1,:,1]
        trace2 = seis1[1,:,2]
        trace3 = seis1[1,:,3]

        prof1 = outputs1[0,:,1]
        prof2 = outputs1[0,:,2]
        prof3 = outputs1[0,:,3]

        pd   = turn(pd)
        gt   = turn(gt)

        Prediction[i*TestBatchSize+k,:,:] = pd
        GT[i*TestBatchSize+k,:,:] = gt
        psnr = PSNR(pd,gt)
        TotPSNR[0,total] = psnr
        ssim = SSIM(pd.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]),gt.reshape(-1,1,label_dsp_dim[0],label_dsp_dim[1]))
        TotSSIM[0,total] = ssim
        # print('The testing psnr: %.2f, SSIM: %.4f ' % (psnr,ssim))
        # total = total + 1

# Save Results
SaveTestResults(TotPSNR,TotSSIM,Prediction,GT,results_dir)
        
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

def plotGT():
   global gnd
   gnd = PlotGroundTruth(GT[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
   return gnd

def plotPD():
    global pred
    pred = PlotPrediction(Prediction[num,:,:],label_dsp_dim,label_dsp_blk,dh,minvalue,maxvalue,font2,font3,SavePath=results_dir)
    return pred

def show_seis():
    global disp
    disp = display_seismic(seis, data_dsp_dim, data_dsp_blk, dh, font2,font3,SavePath=results_dir)
    return disp

def show_trace():
    global trace
    trace = plot_trace(trace1, trace2, trace3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)
    return trace

def show_profile():
    global prof
    prof = plot_profile(prof1, prof2, prof3, data_dsp_dim, data_dsp_blk, dh, font2, font3, SavePath=results_dir)
    return prof

plotGT()
plotPD()
show_seis()
show_trace()
show_profile()

# Record the consuming time
time_elapsed = time.time() - since
# print('Testing complete in  {:.0f}m {:.0f}s' .format(time_elapsed // 60, time_elapsed % 60))

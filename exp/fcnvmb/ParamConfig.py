# -*- coding: utf-8 -*-
"""

"""
####################################################
####             MAIN PARAMETERS                ####
####################################################
SimulateData  = False          # If False denotes training the CNN with SEGSaltData
ReUse         = False         # If False always re-train a network 
DataDim       = [200,301]    # Dimension of original one-shot seismic data
data_dsp_blk  = (1,1)         # Downsampling ratio of input
ModelDim      = [201,301]     # Dimension of one velocity model
label_dsp_blk = (1,1)         # Downsampling ratio of output
dh            = 10            # Space interval 


####################################################
####             NETWORK PARAMETERS             ####
####################################################
if SimulateData:
    Epochs        = 3500       # Number of epoch
    TrainSize     = 1000      # Number of training set
    TestSize      = 100       # Number of testing set
    TestBatchSize = 10
else:
    Epochs        = 500
    TrainSize     = 120      
    TestSize      = 10       
    TestBatchSize = 1


BatchSize         = 3        # Number of batch size
LearnRate         = 0.001      # Learning rate
Nclasses          = 1         # Number of output channels
Inchannels        = 10        # Number of input channels, i.e. the number of shots
SaveEpoch         = 20        
DisplayStep       = 2         # Number of steps till outputting stats

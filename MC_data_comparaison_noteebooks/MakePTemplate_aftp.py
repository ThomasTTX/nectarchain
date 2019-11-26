
##import calin.iact_data.telescope_data_source

## ctapipe
from ctapipe_io_nectarcam import NectarCAMEventSource

import numpy as np

import matplotlib.pyplot as plt
import argparse
import logging
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.optimize import minimize

###from NSB_vbis import dump_pedestals
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,inset_axes
import pandas as pd

from scipy.optimize import curve_fit
from utils import *
import math 
from scipy import stats, optimize, interpolate
import astropy.units as u
import pandas as pd
import seaborn as sns

n_sample =60

def rebin(samples,n_rebin):
    rb_samples=[]
    samples = samples.astype(float)
    for sub_sample in range((len(samples)*n_rebin)-n_rebin):
        rb_samples.append(samples[sub_sample//n_rebin]+(sub_sample%n_rebin)/n_rebin*(samples[(sub_sample//n_rebin)+1]-samples[(sub_sample//n_rebin)]))
        #print(type((samples[sub_sample//n_rebin]-samples[(sub_sample//n_rebin)+1])))
    for sub_sample in range(n_rebin):
        rb_samples.append(samples[len(samples)-1])
    return rb_samples

def splitsum(t1,t2,shift):
    tmpf = UnivariateSpline(np.array(range(len(t2)))+shift,t2,ext=2,k=1,s=0)
    ### DAm bug in UnivariateSpline.intergal assume ext=0 even if not..
    
    #~ ttmp = [tmpf.integral(t,t+1) + t1_val for t,t1_val in enumerate(t1)]
    ttmp =[]
    pp =0
    pp0 =0
    for t,t1_val in enumerate(t1):
        #print(tmpf.integral(t,t+1))
        try :
            pp = tmpf(t+1)
            pp0 = tmpf(t)
            tttmp = tmpf.integral(t,t+1) + t1_val
        except ValueError:
            tttmp = t1_val + pp
            
        ttmp.append(tttmp)
    #plt.plot(tmpf(np.arange(0,60*4+50,1)))
    #plt.show()
    return ttmp

def GetSyncSpePulseShape_aftp(all_norm_wfs,n_rebin=4,ManualShift = 0):
    
    pulseshape = np.zeros(n_sample*n_rebin)

    shifts=[]
    for i_evt,wf in enumerate(all_norm_wfs):

        maxref_0 = 30
        max_wf = np.argmax(wf)
        s_start  = max_wf - 1
        s_end    = max_wf + 2
        s_length = s_end - s_start
        #ped_value = sum(wf[1:11])/10

        try:
            tom0 = sum(np.array(range(s_start,s_end))*np.array(wf[s_start:s_end]))/sum(wf[s_start:s_end])
            shift = (maxref_0*n_rebin)-tom0*n_rebin
            shifts.append(tom0-maxref_0)
        except ValueError:
            print("Value_err")
            continue
        #~ shape = np.clip((rebin((wf-ped_value)/charge,n_rebin)),0,1e6)
        QQ = wf[max_wf-6:max_wf+10].sum()/58.
        shape = rebin((wf/QQ),n_rebin)
        ## shift = int(round(round(maxref_0*n_rebin)-tom0*n_rebin))
           
        # ~ if tom0<10 or tom0>40:
            # ~ print("wah")
            # ~ continue
           
        
        
        pulseshape = splitsum(pulseshape,shape,shift)
        #print(np.argmax(ttt))

    t_0 = np.argmax(pulseshape)/float(n_rebin)
    print('T_0 : {} ns'.format(t_0))
    rebins = np.linspace(0,n_sample-(1/n_rebin),n_sample*n_rebin)
    
    ##ManualShift is here to set the TOM at 30 nanosec (or any val)
    ## Qamp is the mean amplitude/Q ratio
    
    ManualShift = np.mean(shifts)
    print("shft",ManualShift)
    print("Tstd :: ",np.std(shifts))
    # ~ pulse_fct = UnivariateSpline(rebins+ManualShift,np.array(pulseshape)*QAmp/np.max(pulseshape),ext=3,k=3,s=0)
    pulse_fct = UnivariateSpline(rebins+ManualShift,np.array(pulseshape),ext=3,k=3,s=0)
    # ~ pulse_fct = UnivariateSpline(rebins,np.array(pulseshape),ext=3,k=3,s=0)
    return pulse_fct, t_0,shifts,np.std(shifts)



    
if __name__ == "__main__":
    file_path = '/media/ttaverni/Transcend/data/dataR1/NectarCAM.Run1535.00[0-9][0-9].fits.fz'
    
    inputfile_reader = NectarCAMEventSource(
        input_url = file_path,
        max_events=5000)
        
    cfg = inputfile_reader.camera_config
    pix_ids = cfg.expected_pixels_id
    ped_wfs = []
    for ii, event in enumerate(inputfile_reader):
        ped_wfs.append(event.r0.tel[event.r0.tels_with_data[0]].waveform[0,pix_ids])
            
    ped_wfs = np.array(ped_wfs).mean(axis=0)
    print("pedestal estimation Done")

    ##########
    file_paths   = ['/media/ttaverni/Transcend/data/dataR1/NectarCAM.Run1537.00[0-9][0-9].fits.fz',\
                    '/media/ttaverni/Transcend/data/dataR1/NectarCAM.Run1538.00[0-9][0-9].fits.fz',\
                    '/media/ttaverni/Transcend/data/dataR1/NectarCAM.Run1539.00[0-9][0-9].fits.fz']
    wfs = []
    # ~ for file_path in file_paths:
    # ~ for iint in range(37,44):
    for iint in [1]:
        
        # ~ file_path = '/media/ttaverni/Transcend/data/dataR1/NectarCAM.Run15{}.00[0-9][0-9].fits.fz'.format(iint)
        # ~ file_path = '/media/ttaverni/Transcend/data/dataR1/NectarCAM.Run1504.00[0-9][0-9].fits.fz'
        file_path = '/media/ttaverni/Transcend/data/dataR1/NectarCAM.Run1502.00[0-9][0-9].fits.fz'
        inputfile_reader = NectarCAMEventSource(
            input_url = file_path,
            max_events=500000)
        wfs = []
        for ii, event in enumerate(inputfile_reader):
                for pid,pix in enumerate(pix_ids):
                    wf = event.r0.tel[event.r0.tels_with_data[0]].waveform[0,pix]
                    nwf = wf - ped_wfs[pid]
                    # ~ if nwf[10:40].sum()/58. > 60.:
                    if nwf[10:40].max()/14. > 80.:
                        wfs.append(nwf)
                if ii%10000 == 0:
                    print(ii)
                if len(wfs)>20000:
                    break
    
    print("waveform selection Done ({} wfs)".format(len(wfs)))
    
    wfs=np.array(wfs)              
    fct,tmax,shfts,Tstd = GetSyncSpePulseShape_aftp(wfs,n_rebin=4,ManualShift = 0)
    # ~ for ii,wf in enumerate(wfs[:5000]): 
        # ~ plt.plot(np.arange(-shfts[ii],-shfts[ii]+70,1)[:60],wf,color='black',alpha=.1)
        
    fff,ax = plt.subplots()
    xx= np.arange(0,60,.25)
    ax.plot(xx,fct(xx-12.0)/ wfs.shape[0],'-')
    ax.grid(True)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("ADC (A.U.)")
    ax.set_title("Pulse Template")
    ax.legend(numpoints=1,loc=2, prop={'size': 10})



    plt.show()


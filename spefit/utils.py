from matplotlib import pyplot as plt

from astropy.table import Table
import astropy.units as u

# ~ from ctapipe.io.nectarcameventsource import NectarCAMEventSource
from ctapipe_io_nectarcam import NectarCAMEventSource 
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay

import numpy as np
import logging



def pixel_index_to_id(pix_index,conf_mods):
    id_module = conf_mods[pix_index  // 7]
    pixel_id    = id_module*7 + pix_index%7
    return pixel_id


def get_all_wfs(filename,max_events=1000000,telid=0):
    inputfile_reader = NectarCAMEventSource(
        input_url = filename,
        max_events=max_events)
    all_wfs = []
    expected_pixs = inputfile_reader.camera_config.expected_pixels_id
    for ii, event in enumerate(inputfile_reader):
        all_wfs.append(event.r0.tel[0].waveform[:,expected_pixs])
        if ii%10000==0:
            print(ii,event.r0.tel[telid].waveform.mean())
    ## shape =  (nevt, ngain, npix, nsample)
    return np.array(all_wfs)


def get_all_wfs_from_reader(inputfile_reader,telid=0):
    
    all_wfs = []
    for ii, event in enumerate(inputfile_reader):
        # ~ if event.r0.tel[telid].trigger_type == 1:
        if event.nectarcam.tel[telid].evt.ucts_trigger_type == 1:
            all_wfs.append(event.r0.tel[telid].waveform)
        if ii%10000==0:
            print("read {} evts wfs mean {}".format(ii,event.r0.tel[telid].waveform.mean()))
    ## shape =  (nevt, ngain, npix, nsample)
    return np.array(all_wfs)
    

    
def get_full_pedestals(filename):
    inputfile_reader = NectarCAMEventSource(
        input_url = filename,
        max_events=10000)
    expected_pixs = inputfile_reader.camera_config.expected_pixels_id
    all_wfs = []
    telid=0
    for ii, event in enumerate(inputfile_reader):
        # ~ if event.r0.tel[telid].trigger_type == 32:
        if event.nectarcam.tel[telid].evt.ucts_trigger_type == 32:
            all_wfs.append(event.r0.tel[telid].waveform[:,expected_pixs])
            if len(all_wfs) > 1000:
                break
    all_wfs = np.array(all_wfs)
    return all_wfs.mean(axis=0)


######################### counters 

def get_ts2_decimal(ts2):
    '''Converts TS2 byte into decimal value in ns'''
    bits = int(ts2)
    if bits & 0x80 != 0:
        # Return two'scomplement
        return bits - (1 << 8)
    return bits


def get_feb_timestamp(counter_value):
    '''Return time-stamp in ns combining PPS, TS1 and TS2 counters'''
    # "counterValue":[GlobalEventCnt, PPS, EventCnt, TS1, TS2_PPS, TS2_TRIG, 0]
    _, pps, _, ts1, ts2_pps, ts2_trig, _ = counter_value
    ns = pps*1e9 + ts1*8 + get_ts2_decimal(ts2_trig) - get_ts2_decimal(ts2_pps)
    return ns
    
## TODO
def get_charges_hists_from_reader(inputfile_reader,maxs,ds_s,ds_e,telid=0):
    for ii, event in enumerate(inputfile_reader):
        wfs = event.r0.tel[telid].waveform[0] ## shape =  (npix, nsample)
    return 1

def ui8_to_uin(ui8_array):
    return np.sum( [ui8<<(ii*8) for ii,ui8 in enumerate(ui8_array)] )

def read_cdts(evt,telid=0):
    evtn = evt.nectarcam.tel[telid].evt
    cdts_ui8 = evtn.cdts_data
    
    #tUInt32 
    eventCounter        = ui8_to_uin(cdts_ui8[0:4]  )
    #tUInt32  
    ppsCounter          = ui8_to_uin(cdts_ui8[4:8]  )
    #tUInt32 
    clockCounter        = ui8_to_uin(cdts_ui8[8:12] )
    #tUInt64 
    uctsTimeStamp       = ui8_to_uin(cdts_ui8[12:20])
    #tUInt64 
    cameraTimeStamp     = ui8_to_uin(cdts_ui8[20:28])
    #tUInt8 
    triggerType         = cdts_ui8[28]
    #tUInt8 
    whiteRabbitStatus   = cdts_ui8[29]
    #tUInt8 
    arbitraryInformation= cdts_ui8[29]
    return np.array([eventCounter,ppsCounter,clockCounter,uctsTimeStamp,cameraTimeStamp,triggerType,whiteRabbitStatus,arbitraryInformation], dtype='int64')
    
def Simple_CamPlot(image,geom,pix_ids,title=""):
    disp1 = CameraDisplay(geom,title=title)
    disp1.add_colorbar()
    disp1.set_limits_minmax(zmin=image.min()*0.98,zmax=image.max()*1.02)
    blankam1 = np.zeros(1855)
    blankam1[pix_ids]=image
    disp1.image = blankam1
    plt.show()
    
def Simple_AxCamPlot(image,geom,pix_ids,title=""):
    f,ax = plt.subplots()
    disp1 = CameraDisplay(geom,title=title,ax=ax)
    disp1.add_colorbar(ax=ax)
    # ~ disp1.set_limits_minmax(zmin=image.min()*0.98,zmax=image.max()*1.02)
    disp1.set_limits_minmax(zmin=image.min(),zmax=image.max())
    blankam1 = np.zeros(1855)
    blankam1[pix_ids]=image
    disp1.image = blankam1
    
    ax.set_ylim(-0.65,0.65)
    ax.set_xlim(-0.6,0.6)
    return ax

    


from matplotlib import pyplot as plt
import os
import numpy as np
import math 
import pandas as pd
import seaborn as sns


## astropy
import astropy.units as u
from astropy.table import Table

## calin
#from cta.nectarcam.feb.I_NmcConfig import NmcConfig

## ctapipe
# ~ from ctapipe.io.nectarcameventsource import NectarCAMEventSource
from ctapipe_io_nectarcam import NectarCAMEventSource 
from ctapipe import utils
from pkg_resources import resource_filename
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay

##scipy
from scipy.optimize import curve_fit, minimize
from scipy import stats, optimize, interpolate

#local
from utils import *






def gaussian(x, mu, sig):
    return (1./(sig*np.sqrt(2*np.pi)))*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def doubleGauss(x,sig1,mu2,sig2,p):
    return p *2 *gaussian(x, 0, sig1) * (x>=0) + (1-p) * gaussian(x, mu2, sig2)

def PMax(r):
    if (np.pi*r**2/(np.pi*r**2 + np.pi - 2*r**2 - 2) <= 1):
        return np.pi*r**2/(np.pi*r**2 + np.pi - 2*r**2 - 2)
    else:
        return 1

def ax(p,res):
    return ((2/np.pi)*p**2-p/(res**2+1))

def bx(p,mu2):
    return (np.sqrt(2/np.pi)*2*p*(1-p)*mu2)

def cx(sig2,mu2,res,p):
    return (1-p)**2*mu2**2 - (1-p)*(sig2**2+mu2**2)/(res**2+1)

def delta(p,res,sig2,mu2):
    return bx(p,mu2)*bx(p,mu2) - 4*ax(p,res)*cx(sig2,mu2,res,p)

def ParamU(p,r):
    return ((8*(1-p)**2*p**2)/np.pi - 4*(2*p**2/np.pi - p/(r**2+1))*((1-p)**2-(1-p)/(r**2+1)))

def ParamS(p,r):
    return (4*(2*p**2/np.pi - p/(r**2+1))*(1-p))/(r**2+1)

def SigMin(p,res,mu2):
    return mu2*np.sqrt((-ParamU(p,res)+(bx(p,mu2)**2/mu2**2))/(ParamS(p,res)))

def SigMax(p,res,mu2):
    return mu2*np.sqrt((-ParamU(p,res))/(ParamS(p,res)))

def sigma1(p,res,sig2,mu2):
    return (-bx(p,mu2)+np.sqrt(delta(p,res,sig2,mu2)))/(2*ax(p,res))

def sigma2(n,p,res,mu2):
    if ((-ParamU(p,res)+(bx(p,mu2)**2/mu2**2))/(ParamS(p,res)) > 0):
        return SigMin(p,res,mu2)+n*(SigMax(p,res,mu2)-SigMin(p,res,mu2))
    else:
        return n*SigMax(p,res,mu2)
    
def doubleGaussConstrained(x,pp,res,mu2,n):
    p = pp*PMax(res)
    sig2 = sigma2(n,p,res,mu2)
    sig1 = sigma1(p,res,sig2,mu2)
    spe = doubleGauss(x,sig1,mu2,sig2,p) 
    return spe
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################

def pdfspe(xx,res,nn):
    aa = res[0]**nn*np.exp(-res[0])/math.factorial(nn)
    bb = np.exp(-(xx-res[3]*nn-res[1])**2  /  (2*(nn*res[4]**2 +res[2]**2)))
    cc = (2*np.pi * (nn*res[4]**2 +res[2]**2))**0.5
    return aa*bb/cc
    

def get_mes(xx,li,ped,sped,g,sg):
    n_pe = 20
    res=[li,ped,sped,g,sg]
    mes=0
    for nnn in range(n_pe):
        mes+=pdfspe(xx,res,nnn)
    return mes



##########################################
def mes2g(xx,li,ped,sped,res,mu2):

    n = 0.715
    #~ pp =  pp_ 
    pp =  pp_ 
    ##pp = 0.001
    allrange = np.linspace(-1000,1000,2000)
    
    mes = np.exp(-li)/1*stats.norm.pdf(allrange,0,sped)
    
    ppp = stats.norm.pdf(np.linspace(-100,100,200),0,sped)
    spe = doubleGaussConstrained(allrange,pp,res,mu2,n)
    gain = np.sum( (allrange)*spe)
    npe = np.convolve(spe,ppp,"same")
 
    ###print(spe.sum(),li,ped,sped,res,mu2)
    mes+=npe* li*np.exp(-li)
    for nnn in range(2,8):
        npe = np.convolve(npe,spe,"same")
        mes+=npe * li**nnn*np.exp(-li)/math.factorial(nnn)
    fff = interpolate.UnivariateSpline(allrange,mes,ext=1,k=3,s=0)
    return fff(xx-ped)
##########################################################    

    


def fit_ses_2g(Q_array,plot):
    n_ = 0.715  
    Q_min = min(Q_array)
    Q_max = max(Q_array)
    n_bin = int((Q_max-Q_min)/1)
    
    spemodel=doubleGaussConstrained
    fitf = mes2g
    p0=[.9,  np.median(Q_array),15.,  .5, 65]
    #p0=[.8,  4000,15.,  .8, 150]
    ##p0=[.8,  4000. ,10.,  .5, 70]
    
    
    #~ spemodel=pdfspe
    #~ fitf = get_mes
    #~ p0=[.1,  np.median(Q_array),  10,   50, 15]
    
    if plot: 
        fig, (ax,axr) = plt.subplots(nrows=2, sharex=True,figsize=(12, 6),gridspec_kw = {'height_ratios':[4, 1]})
        fig.subplots_adjust(hspace=0)
        n,bins,patches = ax.hist(Q_array,n_bin,range=(Q_min,Q_max), density=0,linewidth=0.3,edgecolor='yellow', facecolor='orange', alpha=0.4,label='spe data')
    else:
        n,bins  = np.histogram(Q_array,bins=n_bin,range=(Q_min,Q_max))
        
    centers = (bins[:-1] + bins[1:]) / 2


    pp,pcov = curve_fit(fitf,centers,n/sum(n),p0=p0,maxfev = 6000,\
            #~ bounds=([0,3000,0,0,0],[2,6000,30,.8,400]))
            bounds=([0,3000,10,.2,10],[4.,7000,70,.7,500]))
            # ~ bounds=([0,3000,10,.2,10],[4.,7000,30,.7,500]))
    ##pp =p0
    ##print("fit :: ",pp)
    xspe = np.linspace(-1000,1000,2000)
    ppped = stats.norm.pdf(np.linspace(-100,100,200),0,pp[2])
    spe = doubleGaussConstrained(xspe,pp_,pp[3],pp[4],n_)
    gain=((xspe)*spe).sum()
    print("=== Gain :: ",gain,'p.e. ------------- ==== Light I',pp[3])
    if plot:
        ax.errorbar(centers, n, yerr = np.array(n)**0.5,fmt='none', color = "orange",alpha = 0.5)
        ax.plot(centers,fitf(centers,*pp)*sum(n),color="red",label='ses model,  Light intesity :: {:.3} p.e.'.format(pp[0]))
        ax.plot(centers,stats.norm.pdf(centers,pp[1],pp[2])*sum(n)*np.exp(-pp[0]),':',color="blue",label='pedestals : mean {:.5} std {:.3}'.format(pp[1],pp[2]))
        #ax.plot(centers,spe*sum(n)*np.exp(-pp[0])*pp[0],'--',label='spe')
        npe=np.convolve(spe,ppped,"same")*sum(n)
        ax.plot(xspe+pp[1],npe*np.exp(-pp[0])*pp[0],'--',label='1 p.e.',alpha=.5)
        for nnn in range(2,5):
            npe = np.convolve(npe,spe,"same")
            ax.plot(xspe+pp[1],npe * pp[0]**nnn*np.exp(-pp[0])/math.factorial(nnn),'--',label='{} p.e.'.format(nnn),alpha=.5)
        
        #~ ff,ax2 = plt.subplots()
        #~ ax2.plot(centers,spe*sum(n)*np.exp(-pp[0])*pp[0],'--',label='spe')
        #~ ax2.plot(centers,np.convolve(spe,ppped,"same")*sum(n)*np.exp(-pp[0])*pp[0],'-',label='Cspe')        
            
        residuals = [(nn2 - fitf(centers[ii],*pp)*sum(n))/(fitf(centers[ii],*pp)*sum(n))**0.5 if nn2<=0 else \
                     (nn2 - fitf(centers[ii],*pp)*sum(n))/nn2**0.5 for ii,nn2 in enumerate(n)]
    

        axr.errorbar(centers,residuals,yerr=1,fmt='.',color='red',alpha=.8)
        
        axr.plot(xspe+pp[1],np.zeros(len(xspe)),':',color='black')
        
        ax.set_title('MES Spetctrum (gain = {:.3} ADC / p.e. )'.format(gain))
        axr.set_xlabel('ADC sum')
        ax.set_ylabel('n')
        axr.set_ylabel('$\sigma$')
        ax.legend(loc="upper right",numpoints=1,ncol=2, prop={'size': 12})
        #ax2.legend(loc="upper right",numpoints=1,ncol=2, prop={'size': 8})
        ax.set_yscale("log", nonposy='clip')
        ax.set_xlim(centers[0]-30,centers[-1])
        ax.set_ylim(1,np.max(n)*1.2)
        axr.set_ylim(-4,4)

    return pp,pcov,gain
    
def plot_allparam_map(df):
    nc = 4
    nr = 2
    f,axs = plt.subplots(ncols=nc,nrows=nr,figsize=(12,6))
    # ~ table = Table.read('./NewNectarCam.camgeom.fits.gz')
    # ~ geom = CameraGeometry.from_table(table)
    # ~ geom.rotate(10.3*u.deg)

    geom = CameraGeometry.from_name("NectarCam-002")
        
    for ii,key in enumerate(df):
    # ~ for ii,key in enumerate(['Light I', 'ped mean', 'ped width', 'res', 'Mu2', 'gain']):
        ax = axs[ii%nr,ii//nr%nc]
        blankam = np.zeros(1855)
        blankam[pix_ids]=df[key]
        
        disp = CameraDisplay(geom,title=key,ax=ax)
        disp.add_colorbar(ax=ax)
        disp.set_limits_minmax(zmin=np.min(df[key])*.99,zmax=np.max(df[key])*1.01)
        disp.image = blankam
        ax.set_ylim(-0.8,0.8)
        ax.set_xlim(-0.5,0.4)
    plt.show()
    return


def get_targeted_pixs(pix0,geom,pix_ids):
    
    targeted_pixs = geom.neighbors[pix0].copy()
    targeted_pixs.append(pix0)
    for pix in targeted_pixs.copy():
        for pixn in geom.neighbors[pix]:
            if (pixn not in targeted_pixs) and (pixn in pix_ids):
                targeted_pixs.append(pixn)
                
            for pixnn in geom.neighbors[pixn]:
                if (pixnn not in targeted_pixs) and (pixnn in pix_ids):
                    targeted_pixs.append(pixnn)

                for pixnnn in geom.neighbors[pixnn]:
                    if (pixnnn not in targeted_pixs) and (pixnnn in pix_ids):
                        targeted_pixs.append(pixnnn)
    return np.array(targeted_pixs)



def get_targeted_res(wfs,pix_ids,plot=False):
    gains_2g= []
    res_2g =  []
    modpix =  []
    module =  []
    maxs = wfs.mean(axis=0).argmax(axis=1)
    
    for pix_index,pix_id in enumerate(pix_ids[:]):
        print("pix ",pix_index,pix_id)
        
        # ~ Q_array1 = all_wfs[:,0,pix_index,mm-6:mm+10].sum(axis=1)
        Q_array1 = wfs[:,pix_index,maxs[pix_index]-6:maxs[pix_index]+10].sum(axis=1)
        
        try:
            res,err,gain = fit_ses_2g(Q_array1,plot=plot)
            
        except:
            res,err,gain = [1,1,1,1,1],1,40
            print("Fit Failed")
            # ~ break
        gains_2g.append(gain)
        res_2g.append(res)
        
        modpix.append(pix_index % 7)
        module.append(pix_index // 7)
        color='black'
        alpha = .1
        
        axt.plot(np.arange(3800,5000,1)-res[1],mes2g(np.arange(3800,5000,1),*res),color=color,alpha=alpha)        
        axspe.plot(np.arange(0,200,1),doubleGaussConstrained(np.arange(0,200,1),pp_,res[3],res[4],0.715),color='black',alpha=.1)


        print("----")

    return np.array(gains_2g),np.array(res_2g),np.array(modpix),np.array(module)



# ~ def get_targeted_wfs(inputfile_reader,pix_indexs):
    # ~ all_wfs = []
    # ~ print("Reading evts for pixs ",pix_indexs)
    # ~ for ii, event in enumerate(inputfile_reader):
        # ~ if event.r0.tel[0].trigger_type == 64:
            # ~ break
        # ~ all_wfs.append(event.r0.tel[0].waveform[0,pix_indexs,:])
        # ~ if ii%10000==0:
            # ~ print(ii,event.r0.tel[0].waveform.mean())
            # ~ print("evt number",event.r0.event_id)
    ## shape =  (nevt, ngain, npix, nsample)
    # ~ return np.array(all_wfs)    

def get_targeted_wfs(inputfile_reader,pix_indexs):
    all_wfs = []
    print("Reading evts for pixs ",pix_indexs)
    ts = 0
    for ii, event in enumerate(inputfile_reader):
        if ii>0 and  event.r0.tel[0].trigger_time-ts > 1e9:
            break
        if event.nectarcam.tel[0].evt.ucts_trigger_type !=32:
            all_wfs.append(event.r0.tel[0].waveform[0,pix_indexs,:])
            ts = event.r0.tel[0].trigger_time
        if ii%10000==0:
            print(ii,event.r0.tel[0].waveform.mean())
            print("evt number",event.r0.event_id)
    return np.array(all_wfs)
    
    
if __name__ == "__main__":
    #~ file_path = "/local/home/ttaverni/cta-nectar/dataR1/NectarCAM.Run0890.0000.fits.fz"
    ares = []
    pp_ = 0.45

    param_names=["Light I","ped mean", "ped width","res", "Mu2"]
    
    #~ file_path = "/local/home/ttaverni/cta-nectar/speR1/NectarCAM.Run0921.0000.fits.fz"
    # ~ file_path = "/media/ttaverni/Transcend/data/2019/NectarCAM.Run1031.0000.fits.fz"
    # ~ file_path = "/media/ttaverni/Transcend/data/2019/NectarCAM.Run1067.0000.fits.fz"
    # ~ file_path = "/media/ttaverni/Transcend/data/2019/NectarCAM.Run1078.0000.fits.fz"
    # ~ file_path = "/local/home/ttaverni/cta-nectar/dataR1/NectarCAM.Run1109.0000.fits.fz"
    # ~ file_path = "/media/ttaverni/Transcend/data/speR1/NectarCAM.Run1126.0000.fits.fz"
    
    # ~ file_path = "/local/home/ttaverni/cta-nectar/spe_target/NectarCAM.Run1178.0000.fits.fz"
    
    #file_path = "/media/ttaverni/Transcend/data/target_spe/NectarCAM.Run0011.0000.fits.fz"
    # ~ file_path = "/local/home/ttaverni/cta-nectar/speR1/NectarCAM.Run1112.0000.fits.fz"
    # ~ file_path = "/media/ttaverni/Transcend/data/target_spe/NectarCAM.Run1180.0000.fits.fz"
    file_path = "/media/ttaverni/Transcend/data/speR1/NectarCAM.Run1554.0000.fits.fz"
    # ~ file_path = "/media/ttaverni/Transcend/data/target_spe/NectarCAM.Run1184.0000.fits.fz"
    # ~ file_path = "/media/ttaverni/Transcend/data/target_spe/NectarCAM.Run0011.0000.fits.fz"
    # ~ file_path = "/media/ttaverni/Transcend/data/target_spe/NectarCAM.Run1184.0000.fits.fz"
    # ~ file_path = "/media/ttaverni/Transcend/data/speR1/NectarCAM.Run1482.0000.fits.fz"
    # ~ file_path = "/media/ttaverni/Transcend/data/speR1/NectarCAM.Run1508.0000.fits.fz"
    # ~ file_path = "/media/ttaverni/Transcend/data/adlersofff/20190524/NectarCAM.Run1247.0000.fits.fz"



    inputfile_reader = NectarCAMEventSource(
        input_url = file_path,
        max_events=1)
    cfg = inputfile_reader.camera_config
    pix_ids = cfg.expected_pixels_id
    
    mmm = 20
    ##########################

    

    # ~ table = Table.read('./NewNectarCam.camgeom.fits.gz')
    # ~ geom = CameraGeometry.from_table(table)
    # ~ geom.rotate(10.3*u.deg) 
        
    geom = CameraGeometry.from_name("NectarCam-002")

    

    ares=[]
    #pix =8

    plot =False
    f,axt = plt.subplots()
    f,axspe = plt.subplots()
    pal = ['#00204C', '#31446B', '#666870', '#958F78', '#CAB969', '#FFE945','red']
    targeted_drawer = [79, 81, 83,118, 116, 114, 112,146, 148, 150, 152,185, 183, 181]
    # ~ targeted_drawer = [135,133,131]
    

    gains_2g= np.ones(len(pix_ids))
    res_2g =  np.array([ np.ones(5) for pix in pix_ids])
    modpix =  np.ones(len(pix_ids))
    module =  np.ones(len(pix_ids))
 
    inputfile_reader = NectarCAMEventSource(
        input_url = file_path,
        max_events=20000)
               
    # ~ for tg in [132]:    
    for tg in targeted_drawer:
        
        
        ### THIS SHOULD BE COMMENTED IF USING TH TARGET
        inputfile_reader = NectarCAMEventSource(
            input_url = file_path,
            max_events=20000)
            
        print("target centred on drawer {}".format(tg))
        targeted_pixs = get_targeted_pixs(tg*7+3,geom,pix_ids)
        
        targeted_pixs_index = np.array([ np.where( pix_ids == pix)[0][0] for pix in targeted_pixs])

        ## we skip pixel wich have their gain estimed
        
        notdone_mask = gains_2g[targeted_pixs_index] == 1
        targeted_pixs =targeted_pixs[notdone_mask]
        targeted_pixs_index =targeted_pixs_index[notdone_mask]        
        
        
        wfs = get_targeted_wfs(inputfile_reader,targeted_pixs)## shape =  (nevt, ngain, npix, nsample)
        tar_gains_2g,tar_res_2g,tar_modpix, tar_module = get_targeted_res(wfs,targeted_pixs_index,plot=False)
    
        gains_2g[targeted_pixs_index] = tar_gains_2g
        res_2g[targeted_pixs_index]   = tar_res_2g
        modpix[targeted_pixs_index]   = tar_modpix
        module[targeted_pixs_index]   = tar_module
    
    ##################
    #### CAM PLOT ####b
    ################## 
    f,axs = plt.subplots(ncols=2,figsize=(12,6))
    blankam1 = np.zeros(1855)
    blankam2 = np.zeros(1855)
    
    blankam2[pix_ids]=gains_2g
    
    # ~ blankam1[pix_ids]=res_2g.T[0]
    # ~ blankam2[pix_ids]=res_2g.T[1]


    disp2 = CameraDisplay(geom,title="gains (2 gauss mes)",ax=axs[1])

    disp2.add_colorbar(ax=axs[1])
    disp2.set_limits_minmax(zmin=gains_2g.min()-5,zmax=gains_2g.max()+5)
    disp2.image = blankam2
    
    

    # ~ pix_HVs     = get_pixs_HV( get_config_xml2(file_path) )
    d= {param_names[0] : res_2g.T[0], \
        param_names[1] : res_2g.T[1], \
        param_names[2] : res_2g.T[2], \
        param_names[3] : res_2g.T[3], \
        param_names[4] : res_2g.T[4], \
        'gain'         : gains_2g   , \
        'pix_num'      : modpix     }
    df = pd.DataFrame(d)
    sns.set()
    disp_p = 0
    
    disp1 = CameraDisplay(geom,title=param_names[disp_p],ax=axs[0])
    disp1.add_colorbar(ax=axs[0])
    disp1.set_limits_minmax(zmin=res_2g.T[disp_p].min()*0.98,zmax=res_2g.T[disp_p].max()*1.02)
    blankam1[pix_ids]=res_2g.T[disp_p]
    disp1.image = blankam1
    axs[0].set_ylim(-0.8,0.8)
    axs[1].set_ylim(-0.8,0.8)
    
    axs[0].set_xlim(-0.5,0.4)
    axs[1].set_xlim(-0.5,0.4)
    

    
    dfff,dax = plt.subplots()
    dax.hist(gains_2g,50,label="mean: {:.4} std : {:.3}".format(gains_2g.mean(),gains_2g.std()))
    dax.legend()
    plt.show()
 
    for ii in range(len(pix_ids)//7):
        print("module {} || {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |".format(pix_ids[ii*7]//7,gains_2g[ii*7],gains_2g[ii*7+1],gains_2g[ii*7+2],gains_2g[ii*7+3],gains_2g[ii*7+4],gains_2g[ii*7+5],gains_2g[ii*7+6]))
        

    mask = abs(gains_2g - (gains_2g.mean())) > 3*gains_2g.std()
    print()
    print(gains_2g[mask])
    yn = input('there is {} pixel whith gain  > 3 sigma (expected $\sim$ {}), do you want more plot ?? (y/b/n)'.format(len(pix_ids[mask]),len(pix_ids)/100))
    if yn=='y'or yn == 'b':
        if yn=='y':
            for pix in np.arange(0,len(targeted_pixs_index),1)[mask]:
                res,err,gain = fit_ses_2g(wfs[:,pix,mmm-8:mmm+16].sum(axis=1),plot=True)
        g = sns.pairplot(df,diag_kind="kde",hue='pix_num',palette=sns.cubehelix_palette(7, start=.5, rot=-.75)) #,kind='reg')
        plot_allparam_map(d)
        plt.show()

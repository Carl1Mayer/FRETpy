from __future__ import print_function
from pylab import *
get_ipython().magic(u'matplotlib inline')
import sys
sys.path.append( "/Users/mayer/Desktop/Research/Fret Measurement/Python/" )
import os
import pandas as pd
import numpy as np
import scipy.integrate
from scipy.optimize import nnls
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
params = {'legend.fontsize': 15,
          'figure.figsize': (14, 6),
         'axes.labelsize': 20,
         'axes.titlesize': 25,
         'xtick.labelsize':15,
         'ytick.labelsize':15,
         'lines.linewidth':2}
matplotlib.rcParams.update(params)


# # Interactive functions

# In[2]:

def plot_spectralsum(spectral_im,x,y,r,exp_name,savedf=False):
    import pylab as pl
    pl.figure(figsize=[10,10])
    pl.plot(x, y, 'w+', mew=2, ms=r,c="black")
    pl.imshow(spectral_im.sum(axis=0))
    d=pd.DataFrame([x,y,r],columns=[exp_name])
    d.index=["x","y","r"]
    if savedf==True:
        d.to_csv("ex_em_spectra_locations/df_"+exp_name+".csv")
    return(pl.show(),d)

def interactiveSpectra(array,radius,name):
    import pylab as pl
    interact(plot_spectralsum,
         spectral_im=fixed(array),
         x=(0,pl.shape(array)[1],1),
         y=(0,pl.shape(array)[2],1),
         r=widgets.IntSlider(min=1,max=20,step=1,value=radius),
         exp_name=name,
         savedf=False,
         __manual=True)


# # Functions for reading images

# In[3]:

def readim(subdir):
    imlist=[]
    for (dirname, dirs, files) in os.walk(os.getcwd()+'/Spectraim_dir/'):
        for filename in files:
        #print(filename)
            if filename.endswith('.tif') :
                imlist.append(tf.TiffFile(filename).asarray())
    imlist=pl.array(imlist)
    return(imlist)

def readim_names(subdir):
    imlist=[]
    for (dirname, dirs, files) in os.walk(os.getcwd()+'/Spectraim_dir/'):
        for filename in files:
        #print(filename)
            if filename.endswith('.tif') :
                imlist.append(filename)
    imlist=pl.array(imlist)
    return(imlist)

def read_spectral_roi(subdir,x,y,r):
    imlist=[]
    for (dirname, dirs, files) in os.walk(os.getcwd()+'/Spectraim_dir/'):
        for filename in files:
        #print(filename)
            if filename.endswith('.tif'):
                t=tf.TiffFile(filename).asarray()[:,y-r/2:y+r/2,x-r/2:x+r/2]
                imlist.append(t)
    imlist=pl.array(imlist)
    return(imlist)

def excitation_emission_cubes_to_spectra(arr):
    new=[]
    for im in range(len(arr)):
        a=arr[im]
        x=a.shape[1]
        y=a.shape[2]
        z=a.shape[0]
        r=pl.reshape(a,(z,x*y))
    
        new.append(r)
    new=pl.array(new)
    return(new)

def average_excitation_spectra(e_arr):
    enew=pl.mean(e_arr,axis=2)
    return(enew)

def rawroi_to_avgspectra(subdir,x,y,r):
    fourDspectra=read_spectral_roi(subdir,x,y,r)
    
    flat_spectra=excitation_emission_cubes_to_spectra(fourDspectra)
    
    avg_spectra=average_excitation_spectra(flat_spectra)
    
    return(avg_spectra)
    
def save_ROI_spectra_perpixel(subdir,x,y,r,exp_name):
    t=read_spectral_roi(subdir,x,y,r)
    e=excitation_emission_cubes_to_spectra(t)
    savedir="ROISpectra_outputs/"+str(exp_name)+"perpixel_"+"x"+str(x)+"_"+"y"+str(y)+"_"+"r"+str(r)+".npy"
    
    return(pl.save(savedir,e))
   


# # Sub-functions for calculating FRET Efficiencies

# In[4]:

#determines fret efficiency based of the unmixing coeficients
def efficiency(coefs,Q_D,Q_A):
    g=(coefs[0]/Q_D)+(coefs[1]/Q_A)
    E=coefs[1]/(Q_A*g)
    return(E)

#unmixes spectra from the donor and acceptor shapes
def unmix (F,D_em,A_em):
    comps=array([D_em,A_em]).T
    A=comps
    coefs,res=nnls(comps,F)
    return(coefs,res)

#removes the direct excitation component from a spectra based on the 
# normalized acceptor excitation and normalization scaling factor for that frequency
def subDirEx (F,ea,Q_A,A_em,gamma):
    return (F-(gamma**-1*ea*Q_A*A_em))

#calculates the acceptor excitation (normalized space)
def calc_ea (alpha,ed21,ExProduct):
    ea21=ed21/ExProduct
    ea1=alpha/((ea21)-(ed21))
    ea2=ea21*ea1
    return(ea1,ea2)

#calculates the alpha parameter over indicies i through j
def calc_alpha (F2n,F1n,ed21,Q_A,A_em,i,j,graph):
    X=F2n-(ed21*F1n)
    alpha_v=X/(A_em*Q_A)
    alpha=np.mean(alpha_v[i:j])
    if graph == 1:
        alphafig, (sub1,sub2) = subplots(1,2)
        sub1.plot(X,label='F2-ed21*F1')
        sub1.plot(A_em,label='Acceptor Emission')
        sub1.set_xlabel('Index')
        sub1.set_ylabel('Normalized Intensity')
        sub1.legend(loc=2)
        sub2.plot(alpha_v, label='alpha')
        sub2.set_xlabel('Index')
        sub2.set_ylabel('Alpha')
        sub2.axhline(alpha,c="red",label="alpha")
        sub2.axvspan(i,j,c="red",label="Averageing Window",alpha=.25)
    return (alpha)

#calculates the product of the excitation ratios based on single fluorophore measurements over indicies i through j
def calc_ExProduct (D1,A1,D2,A2,i,j):
    ExProduct=np.mean(((D2/D1)*(A1/A2))[i:j])
    return (ExProduct)

#calculates donor excitability ratio (ed2/ed1) over indicies i through j
def calc_ed21(F2n,F1n,i,j,graph):
    ed21=np.mean((F2n/F1n)[i:j])
    if graph==1:
        ed21fig, sub1 = subplots(1,1)
        ed21fig.suptitle('Ratio of normalized F2 and F1 emissions', fontsize=25)
        sub1.axvspan(i,j,c="red",label="Averageing Window",alpha=.25)
        sub1.axhline(ed21,c="red",label="Normalized Donor Excitability Ratio")
        sub1.plot(F2n/F1n)
        sub1.set_xlabel('Index')
        sub1.set_ylabel('F2norm/F1norm')
        sub1.legend(loc=2)
        ylim(ed21-1,ed21+1)
    return (ed21)

#normalizes a vector to unit area    
def norm(F):
    gamma=1.0/(scipy.integrate.trapz(F))
    N=F*gamma
    return (N,gamma)

#subtracts a constant such that the background intensity is 0
def subBG (F):
    BG=np.mean(F[0:4])
    return (F-BG)


# # Function for calculating FRET efficiency

# In[5]:

#calculates FRET efficiency based on experimentally determined spectra of the fretting construct and two individual
#fluorophore standards at two excitation frequencies
def calcE (Fret1,Fret2,Donor1,Donor2,Acceptor1,Acceptor2,Q_D,Q_A,graph):
    span=np.shape(Fret1)[0]
    
    Fret1=subBG(Fret1)
    Fret2=subBG(Fret2)
    Donor1=subBG(Donor1)
    Donor2=subBG(Donor2)
    Acceptor1=subBG(Acceptor1)
    Acceptor2=subBG(Acceptor2)
    
    Fret1n,gamma1=norm(Fret1)
    Fret2n,gamma2=norm(Fret2)
    Donor1n=norm(Donor1)[0]
    Donor2n=norm(Donor2)[0]
    Acceptor1n=norm(Acceptor1)[0]
    Acceptor2n=norm(Acceptor2)[0]
    
    if graph==1:
        Spectrafig, sub1 = subplots(1,1)
        Spectrafig.suptitle('Input Spectra', fontsize=25)
        sub1.plot(Fret1,label='FRET Spectra 1',color='green',ls='-')
        sub1.plot(Fret2,label='FRET Spectra 2',color='green',ls='--')
        sub1.plot(Donor1,label='Donor Spectra 1',color='blue',ls='-')
        sub1.plot(Donor2,label='Donor Spectra 2',color='blue',ls='--')
        sub1.plot(Acceptor1,label='Acceptor Spectra 1',color='orange',ls='-')
        sub1.plot(Acceptor2,label='Acceptor Spectra 2',color='orange',ls='--')
        sub1.set_xlabel('Emission Wavelength (nm)')
        sub1.set_ylabel('Intensity')
        sub1.legend(loc=1)
    
    A_em=(Acceptor1n+Acceptor2n)/2
    D_em=(Donor1n+Donor2n)/2
    
    ed21=calc_ed21(Fret2n,Fret1n,int(0.5*span),int(0.6*span),graph)
    
    exprod=calc_ExProduct(Donor1,Acceptor1,Donor2,Acceptor2,int(0.65*span),int(0.85*span))
    
    alpha=calc_alpha(Fret2n,Fret1n,ed21,Q_A,A_em,int(0.65*span),int(0.85*span),graph)

    ea1,ea2=calc_ea(alpha,ed21,exprod)
    
    Fret1corr=subDirEx(Fret1,ea1,Q_A,A_em,gamma1)
    Fret2corr=subDirEx(Fret2,ea2,Q_A,A_em,gamma2)
    
    coefs1,res1=unmix(Fret1corr,D_em,A_em)
    coefs2,res2=unmix(Fret2corr,D_em,A_em)
    
    E1=efficiency(coefs1,Q_D,Q_A)
    E2=efficiency(coefs2,Q_D,Q_A)
    
    return (E1,E2,res1,res2)


# # Funtions for calcualating the uncertainty based on experimental noise

# In[6]:

def addnoise (F,m,b):
    Fn=F+(m*F**.5+b)*np.random.normal(0,1,F.shape)
    return Fn


# In[7]:

def AveEnsemble (Fret1,Fret2,Donor1,Donor2,Acceptor1,Acceptor2,Q_D,Q_A,n,pixmax,m,b):
    span=Fret1.shape[0]
    arraySpec=np.zeros([2,span,pixmax,n])
    for i in range (0,pixmax):
        for j in range (0,n):    
            arraySpec[0,:,i,j]=Fret1
            arraySpec[1,:,i,j]=Fret2
    arraySpec=addnoise(arraySpec,m,b)
    
    
    steps=100
    pixstep=np.logspace(0,np.log10(pixmax),steps)
    aveSpec=np.zeros([2,span,steps,n])
    aveSpecE=np.zeros([4,steps,n])
    for i in range (0,steps):
        for j in range (0,n):
            if np.isnan(aveSpecE[0,i-1,j-1] or aveSpecE[0,i-1,j-1]):
                aveSpec[0,:,i,j]=np.mean(arraySpec[0,:,0:int(pixstep[i-1]),j],axis=1)
                aveSpec[1,:,i,j]=np.mean(arraySpec[1,:,0:int(pixstep[i-1]),j],axis=1)
                aveSpecE[:,i,j]=calcE(aveSpec[0,:,i,j],aveSpec[1,:,i,j],
                                       Donor1,Donor2,
                                       Acceptor1,Acceptor2,
                                       Q_D,Q_A,0)
    
    AveFig, sub1 = subplots(1,1)
    sub1.set_xlabel('Number of Pixels')
    sub1.set_ylabel('FRET Efficiency')
    sub1.plot(pixstep,aveSpecE[0,:,:],color='blue',ls='-',alpha=(2.0/n))
    sub1.plot(pixstep,aveSpecE[1,:,:],color='green',ls='-',alpha=(2.0/n))

    sub1.plot(pixstep,np.nanmean(aveSpecE[0,:,:],axis=1),color='blue',ls='-',label='E1 of Averaged Spectra')
    sub1.plot(pixstep,np.nanmean(aveSpecE[1,:,:],axis=1),color='green',ls='-',label='E2 of Averaged Spectra')

    sub1.legend(loc=1,prop={'size':12})
    sub1.set_xscale('log')
    xlim(1,pixmax)
    ylim(0,1)
    
    return (aveSpecE,pixstep)


# # Functions for filter based FRET measurements

# In[8]:

#Spectra contains all spectra shape over the same wavelength range
#Donor_ex=Spectra[0,:]
#Donor_em=Spectra[1,:]
#Acceptor_ex=Spectra[2,:]
#Acceptor_em=Spectra[3,:]
#Intensity=Spectra[4,:]
#Filter_ex=Spectra[5,:]
#Filter_m=Spectra[6,:]
#Filter_D=Spectra[7,:]
#Filter_A=Spectra[8,:]
    
def FRET2D (Spectra,eff,qd,QD,QA):
    span=Spectra.shape[1]
    FRET=np.zeros([span,span])
    for i in range (0,span):
        for j in range (0,span):
            FRET[i,j]=Spectra[4,i]*(qd*Spectra[0,i]*((1-eff)*Spectra[1,j]*QD+eff*Spectra[3,j]*QA)
                            +(1-qd)*Spectra[2,i]*Spectra[3,j]*QA)
    return (FRET)
            
def PixRatio (FRET,Spectra,noise,noiseparam,graph):    
    span=Spectra.shape[1]
    if noise==1:        
        FRET=addnoise(FRET*noiseparam[0],noiseparam[1],noiseparam[2])
    
    FRET_f=np.zeros([span,span])
    FRET_f_D=np.zeros([span,span])
    FRET_f_A=np.zeros([span,span])

    for j in range (0,span):
        FRET_f[:,j]=FRET[:,j]*Spectra[5,:]*(1-Spectra[6,:])
    for i in range (0,span):
        FRET_f_D[i,:]=FRET_f[i,:]*Spectra[7,:]*Spectra[6,:]
        FRET_f_A[i,:]=FRET_f[i,:]*Spectra[8,:]*Spectra[6,:]
    
    Ratio=np.sum(FRET_f_D)/np.sum(FRET_f_A)
    
    if graph==1:
        Standardfig, (sub1,sub2) = subplots(1,2)
        Standardfig.suptitle('Filtered Spectra', fontsize=25)
        sub1.imshow(FRET_f_D)
        sub1.set_xlabel('Donor Emission')
        sub1.set_ylabel('Donor Excitation')
        sub2.imshow(FRET_f_A)
        sub2.set_xlabel('Acceptor Emission')
        sub2.set_ylabel('Acceptor Excitation')
        
    return(Ratio)

def Calc_qd(Spectra,expR,eff,QD,QA,points,graph):
    qd=np.linspace(0,1,points)
    span=Spectra.shape[1]
    Calib=np.zeros([span,span,points])
    Ratio=np.zeros([points])
    for k in range (0,points):
        Calib[:,:,k]=FRET2D(Spectra,eff,qd[k],QD,QA)
        Ratio[k]=PixRatio(Calib[:,:,k],Spectra,0,0,0)            
    
    qd_exp=np.interp(expR,Ratio,qd)
    
    if graph==1:
        Standardfig, sub2 = subplots(1,1)
        Standardfig.suptitle('Excitation Ratio Calibration', fontsize=25)
        sub2.plot(qd,Ratio,color='blue')
        sub2.set_xlabel('Donor Excitation Fraction (qd/(qd+qa))')
        sub2.set_ylabel('Intensity Ratio (Donor/Acceptor)')
        sub2.axhline(expR,color='red',ls='--')
        sub2.axvline(qd_exp,color='red',ls='--')
        sub2.legend(loc=4)
    
    return (qd_exp)

def CalcFRET(Spectra,expR,qd,QD,QA,points,graph):
    eff=np.linspace(0,1,points)
    span=Spectra.shape[1]
    Calib2=np.zeros([span,span,points])
    Ratio=np.zeros([points])
    for m in range (0,points):
        Calib2[:,:,m]=FRET2D(Spectra,eff[m],qd,QD,QA)
        Ratio[m]=PixRatio(Calib2[:,:,m],Spectra,0,0,0)            
    
    eff_exp=np.interp(expR,np.flipud(Ratio),np.flipud(eff))
    
    if graph==1:
        Standardfig, sub2 = subplots(1,1)
        Standardfig.suptitle('FRET Efficiency Lookup', fontsize=25)
        sub2.plot(eff,Ratio,color='blue')
        sub2.set_xlabel('FRET Efficiency')
        sub2.set_ylabel('Intensity Ratio (Donor/Acceptor)')
        sub2.axhline(expR,color='red',ls='--')
        sub2.axvline(eff_exp,color='red',ls='--')
        sub2.legend(loc=4)
    
    return (eff_exp)


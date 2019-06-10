from math import ceil
from gwpy.timeseries import TimeSeries
import numpy as np

#########################################################################################################

#########################################################################################################

#########################################################################################################


def BLWNB(f,df,dt,fs):
    #BLWNB - Generate a random signal of given duration with constant power
    #in a given band (and zero power out of band).

    #   x = blwnb(f,df,dt,fs)
    #
    #   f   Scalar. Minimum signal frequency [Hz].
    #   df  Scalar. Full signal bandwidth [Hz].
    #   dt  Scalar. Signal duration [s].
    #   fs  Scalar. Signal sample rate [Hz].

    # Power is restricted to the band (f,f+df). 
    # Note that fs must be greater than 2*(f+df).

    # original version: L. S. Finn, 2004.08.03

    # $Id: BLWNB.m 4992 2015-07-25 18:59:12Z patrick.sutton@LIGO.ORG $

    #% ---- Check that fs > 2*(f+df), otherwise sampling rate is not high enough to
    #       cover requested frequency range of the signal.
    if (fs <= abs(2*(f+df))):
        raise ValueError('Sampling rate fs is too small, fs = '+str(fs)+' must be greater than 2*(f+df) = '+str(np.abs(2*(f+df))))


    if f < 0 or df <= 0 or fs <= 0 or dt <= 0 :
        raise ValueError('All arguments must be greater than zero')

    #% ---- Generate white noise with duration dt at sample rate df. This will be
    #%      white over the band [-df/2,df/2].

    nSamp = ceil(dt*df)
    x_old = TimeSeries(np.random.randn(nSamp),sample_rate=1/dt)
    


    #% ---- Resample to desired sample rate fs.
    x=x_old.resample(fs/df)
    #frac = Fraction(Decimal(fs/df))
    #p, q = frac.numerator , frac.denominator


    #% ---- Note that the rat() function returns p,q values that give the desired
    #%      ratio to a default accuracy of 1e-6. This is a big enough error that
    #%      x may be a few samples too short or too long. If too long, then truncate
    #%      to duration dt. If too short, zero-pad to duration dt.
    #print((np.zeros(nSamp-len(x)).shape))
    nSamp = round(dt*fs)
    if len(x) > nSamp:
        x = x[0:nSamp]
    elif len(x) < nSamp:
        x=np.hstack((np.array(x),np.zeros(nSamp-len(x))))

    #% ---- Heterodyne up by f+df/2 (moves zero frequency to center of desired band).
    fup = f+df/2.
    x = x*np.exp(-2*np.pi*1j*fup/fs*np.arange(1,len(x)+1))

    #% ---- Take real part and adjust amplitude.
    x = np.array(np.real(x)/np.sqrt(2))
    #% ---- Done.
    return(x)


def envelope(t,q=0.5,t0='default',sig=3):
    
    T=t[-1]-t[0]
    length=len(t)
    fs=int(length/T)
    
    if q>=1-1/fs: q=1-1/fs
    if q<=1/fs: q=1/fs

    sigma=T/sig
    sigma_m=q*sigma
    sigma_p=(1-q)*sigma
    

    
    if t0=='default':
        t0=t[0]+sig*sigma_m
        
    tm=np.arange(0,t0,1/fs)
    tp=np.arange(t0,t[-1],1/fs)        

    
    env_m=np.exp(-((tm-t0)/(sigma_m))**2)
    env_p=np.exp(-((tp-t0)/(sigma_p))**2)
    envel=np.hstack((env_m,env_p))

    
    if (len(envel)>length):
        envel=np.delete(envel,-1)

    elif (len(envel)<length):
        envel=np.append(0,envel)


    return(envel)


def decimal_power(t,n):
    if t[0]<0:
        for i in range(len(t)):
            if t[i]>0:
                pos_index=i
                break
        yp=t[pos_index:]**n
        yn=-(-t[:pos_index])**n
        y=np.append(yn,yp)
    else:
        y=t**n
    return(y)

def sig(t,T0,step):
    #T0: time of center of sigmoid
    #step: the duration of the step.
    y=-1/(1+np.exp((10/step)*(t-T0)))+1
    return(y)


#########################################################################################################
#########################################################################################################
#########################################################################################################






def ringdown(f0          # Frequency
             ,h_peak=1   # Initial amplitude
             ,tau=None   # Dumping time tau_min=0.8/f0, tau_max=0.14 for T=1s 
             ,T=None     # Duration of output signal
             ,t0=0       # Time of the peak
             ,delta=np.random.rand()*2*np.pi    # Phase
             ,fs=2048):  # Frequency sample

    
    if tau==None: tau=1/f0    
    if T==None: T=1
        
    sigma=1/f0/4 # This sigma value creates a half gaussian at the beggining of 4 sigma.

    t_m=np.arange(t0-1/f0,t0,1/fs)
    t_p=np.arange(t0,T+t0-1/f0-1/fs,1/fs)
    
    h_p = h_peak*np.cos(2*np.pi*(t_p-t0)*f0+delta)*np.exp(-(t_p-t0)/tau)            #t>=t0
    h_m = h_peak*np.cos(2*np.pi*(t_m-t0)*f0+delta)*np.exp(-0.5*((t_m-t0)/sigma)**2) #t<t0

    h=np.append(h_m,h_p)
    t=np.append(t_m,t_p)

    return(t,h)

def WNB(param,T,fs,q='gauss',seed=np.random.randint(0,1e3)):#param=[h_rss, fc, df]

    # ---- Turn off default interpolation.
    #      (Might actually be useful here, but don't want to alter
    #      frequency content of noise by low-pass filtering).
    #pregen = 0
    h_rss=param[0]
    fc=param[1]
    df=param[2]
    
    t=np.arange(0,T,1/fs)

    # ---- Gaussian-modulated noise burst, white over specified band.

    #% ---- If fifth parameter is specified, use it to set the seed for
    #%      the random number generator.
    #%      KLUDGE: injections will be coherent only if all the
    #%      detectors have the same sampling frequency.

    #if(length(parameters)>=5)
    #    randn('state',parameters(5));
    #end

    #% ---- Gaussian-like envelope
    if q=='gauss':
        env = envelope(t,0.5,t0='defult')
    elif q=='noenv':
        env = np.ones(len(t))
    elif (isinstance(q,float) or isinstance(q,int)) == True:
        env = envelope(t,q,t0='defult')

    #% ---- Band-limited noise (independent for each polarization)
    x = BLWNB(max(fc-df,0),2*df,T,fs)
    h = h_rss*env*x
    
    return(t,h)



def chirplet(T,fs
              ,t0=0
              ,f0=20+np.random.rand()*30
              ,fe=50+np.random.rand()*(250)
              ,phi=np.random.rand()*2*np.pi
              ,n='default'
              ,tc='default'
              ,wnb_envelope=True
              ,ENV='single'
              ,demo=False):



    # ENVELOPE FORM

    ## WNB FORM

    if wnb_envelope==True:
        
        wnb_fc = 2+np.random.rand()*10
        wnb_df = 1+np.random.rand()+5
        t,wnb=WNB(param=[1, wnb_fc, wnb_df ],T=T,fs=fs,q='noenv')
    else:
        t,wnb=np.arange(0,T,1/fs), -0.5*np.ones(fs)


    ## EDGE SMOOTHING OUTTER ENVELOPE
    if ENV=='double':
        q1=0.2+np.random.rand()*0.6
        q2=0.2+np.random.rand()*0.6
        
        sig1=2.5+np.random.rand()*2
        sig2=2.5+np.random.rand()*2

        a1=0.2+np.random.rand()*0.8
        a2=0.2+np.random.rand()*0.8

        e1=a1*envelope(t,q=q1,t0='default',sig=sig1)
        e2=a2*envelope(t,q=q2,t0='default',sig=sig2)

        env=(wnb+1.5)*(e1+e2)

    elif ENV=='single':
        q1=0.1+np.random.rand()*0.8
        env=(wnb+1.5)*envelope(t,q=q1,t0='default',sig=2.5+np.random.rand()*2)

    # FREQUENCY FUNCTION

    if n=='default':
        token=np.random.randint(0,2)

        ## Power 0 < n <1
        if (-1)**token == -1:
            n = 0.1+np.random.rand()*0.9
        ## Power 1 < n < 5
        elif (-1)**token == 1:
            n = 1+np.random.rand()*4

    if tc=='default':
        tc = 0.2+np.random.rand()*0.6

    f=f0+((fe-f0)/((1-tc)**n+tc**n))*decimal_power(t-tc,n)+((fe-f0)*(tc**n)/((1-tc)**n+tc**n))



    # FINAL BURST INJECTION
    s=env*np.cos(2*np.pi*f*(t-t0)+phi)   

    if demo==True:
        fig=plt.figure(figsize=(15,7))

        gs = GridSpec(2,3, figure=fig)
        
        ax0=fig.add_subplot(gs[0,0:2])
        ax0.plot(t,s,'royalblue')
        ax0.set_title('Waveform Timeseries')
        ax0.set_xlabel('Time')
        ax0.set_ylabel('Amplitude')

        ax0s=fig.add_subplot(gs[1,0:2])
        ax0s.loglog(np.fft.fftfreq(len(s),1/fs)[0:int(fs/2)],np.abs(np.fft.rfft(s)[:-1]),'darksalmon')
        ax0s.set_xlim(20,1024)
        ax0s.set_title('Waveform FFT')
        ax0s.set_xlabel('Frequency')
        ax0s.set_ylabel('Amplitude')
        
        ax1=fig.add_subplot(gs[0,2])
        ax1.plot(t,f,'blueviolet')
        ax1.set_title('Frequency change function')
        ax1.set_ylabel('Frequency')
        ax1.set_xlabel('Time')

        
        ax2=fig.add_subplot(gs[1,2])
        ax2.plot(t,env,'g')
        ax2.set_title('Envelope function')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Amplitude')        
        if wnb_envelope==True:
            ax2.plot(t,wnb+1.5)
    else:
        return(t,s)


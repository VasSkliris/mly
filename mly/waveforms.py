from math import ceil
from gwpy.timeseries import TimeSeries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#########################################################################################################

#########################################################################################################

#########################################################################################################

def envelope(strain
             ,option=None
             ,**kwargs):
    
    """Envelope is a wrapper function that covers all types of envelopes available here.
    
        Arguments
        ---------
        
        strain: list/array/gwpy.TimeSeries
            The timeseries to be enveloped
            
        option: {'sigmoid'} (optional)
            The type of envelope you want to apply. If not specified it defaults to 'sigmoid'
        
        **kwargs: Any keyword arguments acompany the option of envelope.
    
        Returns
        -------
        numpy.ndarray
            The same timeseries that had in the input, but enveloped.
    
    
    """
    
    
    if option == None: option='sigmoid'
        
    if 'fs' in kwargs:
        if not (isinstance(kwargs['fs'],int) and kwargs['fs'] >= 1):
            raise ValueError('fs must be greater than 2*fax')
        else:
            fs=kwargs['fs']
            duration=len(strain)/fs
    else:
        fs=len(strain) 
        duration=1
        
    if option == 'sigmoid':
        
        envStart=sigmoid(np.arange(int(len(strain)/10)))
        envEnd=sigmoid(np.arange(int(len(strain)/10)),ascending=False)
        env=np.hstack((envStart,np.ones(len(strain)-len(envStart)-len(envEnd))-0.01,envEnd))
                
    if option == 'wnb':
        

            
        envStart=sigmoid(np.arange(int(len(strain)/10)))
        envEnd=sigmoid(np.arange(int(len(strain)/10)),ascending=False)
        
        fmin = 1
        fmax = fmin+1+np.random.rand()*10
        wnb=1.2+np.random.rand()*1.8+WNB(duration=duration,fs=fs,fmin=fmin,fmax=fmax)
        env=wnb*np.hstack((envStart,np.ones(len(strain)-len(envStart)-len(envEnd))-0.01,envEnd))
        env=env/np.abs(np.max(env))
        
    return env*np.array(strain)


def sigmoid(timeRange
            ,t0=None
            ,stepTime=None
            ,ascending=True):
    
    """Sigmoid is a functIon of a simple sigmoid.

        Parameters
        ----------
        
        timeRange: list/numpy.ndarray 
            The time range that the sigmoid will be applied.
            
        t0: int/float (optional)
            The time in the center of the sigmoid. If not specified,
            the default value is the center of the timeRange.
        
        stepTime: int/float (optional)
            The time interval where the function will go from 0.01 
            to 0.99 (or the oposite). If not specified, the default value 
            is the duration of the timeRange.
        
        ascending: bool (optional)
            If True the sigmoid will go from 0 to 1, if False from 1 to 0.
            If not specified the default value is True.
            
        Returns
        -------
        
        numpy.ndarray
            A sigmoid function
    """
    if t0 == None: t0=(timeRange[-1]-timeRange[0])/2
    if stepTime == None: stepTime=(timeRange[-1]-timeRange[0])
    if ascending == None: ascending == True
    
    a=((-1)**(int(not ascending)))*(np.log(0.01/0.99)/(stepTime-t0))
    
    y=1/(1+np.exp(a*(np.array(timeRange)-t0)))
    return y

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


def old_envelope(applicationTime # Time that the envelope will be applied
             ,fs             # Sample Frequency
             ,q=0.5          # Weight of the two gaussians creating the envelope
             ,t0='default'   # Max value time of envelope
             ,sig=3          # Combined sigma of two gaussians
             ,duration=None):# Duration of the combined gaussians
    

    # Duration of the signal is smaller that the duration of the envelope
    if isinstance(duration,(float,int)) and duration < applicationTime :
        T = duration
        pad = np.zeros(int((applicationTime-duration)*fs/2))
    else:    # Envelope time = duration (default)
        T = applicationTime
    
            
    
    t=np.arange(0,T,1/fs)
    
    length=len(t)
    
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
        
        
    # Envelope time = duration (default)
    
    if isinstance(duration,(float,int)) and duration < applicationTime :
        envel = np.hstack((pad,envel,pad))
        
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


#########################################################################################################
#########################################################################################################
#########################################################################################################




def csg(frequency 
        ,duration 
        ,phase=None   
        ,fs=None
        ,sigma=None
        ,ellipticity=None
        ,amplitude=None):
    
    """
    Parameters
    ----------

    frequency: int/float
        The frequency of the oscilator.

    duration: int/float 
        The duration of the final timeseries.

    phase: int/float - (0,2π] , optional 
        The phase of the oscilator. If not specified the default is a random phase.

    fs: int , optional
        The sample frequency of the timeseries to be generated. If not specified 
        the default is 1.
        
    sigma: int/float, optional
        The standar deviation of the gaussian envelope. If not specified the default
        is 5. This scales along with duration, so don't use smaller numbers to avoid
        edge cut effects.
    
    ellipticity: int/float
        The ellipticity of the system. It will modify the cross polarisation.
    amplitude: int/float , optional
        The maximum amplitude of the singal. If not specied 
        the default is 1.

    Returns
    -------
    numpy.ndarray
        A numpy array with size fs*duration

    Notes
    -----

    If you want to generate many of those waveforms, the main parameters you need to 
    loop around are frequency, duration and phase.

    """
    if not frequency >=0:
        raise ValueError('Frequency must be a positive number')
        
    if phase == None:
        phase = np.random.rand()*2*np.pi
    elif not (isinstance(phase,(int,float)) and 0<=phase<=2*np.pi):
        raise ValueError('phase must be a number between zero and 2pi.')
        
    if fs == None:
        fs=1
    elif not (isinstance(fs,int) and fs>=1):
        raise ValueError('sample frequency must be an integer bigger than 1.')    
    
    if sigma == None: sigma = 5
    if not (sigma > 0):
        raise ValueError('Sigma must be positive')
        
    if ellipticity==None:
        ellipticity=0
    if not (isinstance(ellipticity,(float,int)) and 0<=ellipticity<=1):
        raise ValueError('Elliptisity must be a numeber in the interval [0,1]')
        
    if amplitude == None:
        amplitude = 1
    elif not (isinstance(amplitude,(int,float))):
        raise ValueError('amplitude must be a number')
         
    t=np.arange(0,duration,1/fs)
    hp=np.sin(2*np.pi*frequency*t+phase)*np.exp(-((t-duration/2)/(duration/sigma))**2)
    hc=np.cos(2*np.pi*frequency*t+phase)*np.exp(-((t-duration/2)/(duration/sigma))**2)*np.sqrt(1-ellipticity**2)
    
    return(hp,hc)


def ringdown(frequency 
             ,duration = None
             ,damping = None   
             ,phase=None 
             ,fs=None
             ,amplitude=None):
    
    """Ringdown is a simple damping oscilator signal. At the beggining of the signal we use a 
    gaussian envelope to make the transition from 0 to maximum amplitude smooth. The signal 
    follows the following formula:
    
    ..math:: hp=A\cos(2\pi ft + \phi)e^{-t/\tau}  \qquad\qquad  \tau = duration/\ln(damping)
    ..math:: hc=A\sin(2\pi ft + \phi)e^{-t/\tau}  \qquad\qquad  \tau = duration/\ln(damping)

    
    Parameters
    ----------

    frequency: int/float
        The frequency of the damping oscilator.

    duration: int/float , optional 
        The duration of the final timeseries. If not specified the defult value is 1.

    damping: int, optional
        Damping describes the fraction of the initial amplitude 
        the signal will have at the end of the timeseries. A damping value of 10 means that
        in the end of the signal the amplitude will be 1/10 of the initial amplitude.
        If not specified the default value is 256.

    phase: int/float - (0,2π] , optional 
        The phase of the oscilator. If not specified the default is a random phase.

    fs: int , optional
        The sample frequency of the timeseries to be generated. If not specified 
        the default is 1.

    amplitude: int/float , optional
        The maximum amplitude of the singal. If not specied 
        the default is 1.

    Returns
    -------
    numpy.ndarray
        A numpy array with size fs*duration

    Notes
    -----

    If you want to generate many of those waveforms, the main parameters you need to 
    loop around are frequency, duration and phase. Changing damping parameter will not give 
    any difference and it is good to keep it big so that you don't have discontinuities at the end.


    """
    if not isinstance(frequency,(int,float)) and frequency >=0:
        raise ValueError('Frequency must be a positive number')
        
    if duration == None:
        duration = 1
    elif not (isinstance(duration,(int,float)) and duration > 1/frequency):
        raise ValueError('duration must be a positive number bigger than 1/frequency.')
        
    if damping == None:
        damping = 256
    elif not (isinstance(damping,(int,float)) and damping >= 1):
        raise ValueError('damping must be a number bigger than or equal to 1.')
    
    if phase == None:
        phase = np.random.rand()*2*np.pi
    elif not (isinstance(phase,(int,float)) and 0<=phase<=2*np.pi):
        raise ValueError('phase must be a number between zero and 2pi.')
        
    if fs == None:
        fs=1
    elif not (isinstance(fs,int) and fs>=1):
        raise ValueError('sample frequency must be an integer bigger than 1.')    
    
    if amplitude == None:
        amplitude = 1
    elif not (isinstance(amplitude,(int,float))):
        raise ValueError('amplitude must be a number,')
         
    
    t0=0
    alpha=np.log(damping)/(duration-1/frequency)
        
    sigma=1/frequency/4 # This sigma value creates a half gaussian at the beggining of 4 sigma.

    t_m=np.arange(t0-1/frequency,t0,1/fs)
    t_p=np.arange(t0,duration+t0-1/frequency-1/fs,1/fs)
    
    h_pp = amplitude*np.cos(2*np.pi*(t_p-t0)*frequency+phase)*np.exp(-(t_p-t0)*alpha)          #t>=t0
    h_mp = amplitude*np.cos(2*np.pi*(t_m-t0)*frequency+phase)*np.exp(-0.5*((t_m-t0)/sigma)**2) #t<t0
    h_pc = amplitude*np.sin(2*np.pi*(t_p-t0)*frequency+phase)*np.exp(-(t_p-t0)*alpha)          #t>=t0
    h_mc = amplitude*np.sin(2*np.pi*(t_m-t0)*frequency+phase)*np.exp(-0.5*((t_m-t0)/sigma)**2) #t<t0
    
    hp=np.append(h_mp,h_pp)
    hc=np.append(h_mc,h_pc)
    t=np.append(t_m,t_p)

    return(hp,hc)


def WNB(duration
      ,fs
      ,fmin
      ,fmax
      ,enveloped=True
      ,sidePad=None):
    
    """Generate a random signal of given duration with constant power
    in a given frequency range band (and zero power out of the this range).
    
        Parameters
        ----------
        
        duration: int/float
            The desirable duration of the signal. Duration must be bigger than 1/fs
        fs: int
            The sample frequncy of the signal        
        fmin: int/float 
            The minimum frequency
        fmax: int/float 
            The maximum frequency
        enveloped: bool (optional)
            If set to True it returns the signal within a sigmoid envelope on the edges.
            If not specified it is set to False.
        sidePad: int/bool(optional)
            An option to pad with sidePad number of zeros each side of the injection. It is suggested
            to have zeropaded injections for the timeshifts to represent 32 ms, to make it easier
            for the projectwave function. If not specified or 
            False it is set to 0. If set True it is set to ceil(fs/32). WARNING: Using sidePad will 
            make the injection length bigger than the duration
            
            
        Returns
        -------
        
        numpy.ndarray
            The WNB waveform
    """
    if not (isinstance(fmin,(int,float)) and fmin>=1):
        raise ValueError('fmin must be greater than 1')
    if not (isinstance(fmax,(int,float)) and fmax>fmin):
        raise ValueError('fmax must be greater than fmin')
    if not (isinstance(fs,int) and fs >= 2*fmax):
        raise ValueError('fs must be greater than 2*fax')
    if not (isinstance(duration,(int,float)) and duration>1/fs):
        raise ValueError('duration must be bigger than 1/fs') 
    if sidePad == None: sidePad=0
    if isinstance(sidePad,bool):
        if sidePad==True:
            sidePad=ceil(fs/32)
        else:
            sidePad=0       
    elif isinstance(sidePad,(int,float)) and sidePad>=0:
        sidePad=int(sidePad)

    else:
        raise TypeError('sidePad can be boolean or int value.'
                        +' If set True it is set to ceil(fs/32).')
        
    df=fmax-fmin
    T=ceil(duration)        
        
        
    # Generate white noise with duration dt at sample rate df. This will be
    # white over the band [-df/2,df/2].
    nSamp = ceil(T*df)
    h=[]
    for _h in range(2):
        x_ = TimeSeries(np.random.randn(nSamp),sample_rate=1/T)

        # Resample to desired sample rate fs.
        x=x_.resample(fs/df)

        # Heterodyne up by f+df/2 (moves zero frequency to center of desired band). 
        fshift = fmin+df/2.
        x = x*np.exp(-2*np.pi*1j*fshift/fs*np.arange(1,len(x)+1))

        # Take real part and adjust length to duration and normalise to 1.
        x = np.array(np.real(x))[:int(fs*duration)]/np.sqrt(2)
        x = x/np.abs(np.max(x))
        h.append(x)
       
    hp=h[0]
    hc=h[1]
    
    if enveloped==True:
        hp=envelope(hp,option='sigmoid')
        hc=envelope(hc,option='sigmoid')
    
    if sidePad!=0:
        hp=np.hstack((np.zeros(sidePad),hp,np.zeros(sidePad)))
        hc=np.hstack((np.zeros(sidePad),hc,np.zeros(sidePad)))

    return(hp,hc)

def chirplet(duration
             ,fs
             ,fmin
             ,fmax
             ,powerLaw=None
             ,powerSymmetry=None
             ,tc=None
             ,phase=None
             ,enveloped=False
             ,returnFrequency=False):
    
    """Generate a random cosine signal of given duration with evolving frequency.
    
        Parameters
        ----------
        
        duration: int/float
            The desirable duration of the signal. Duration must be bigger than 1/fs
        fs: int
            The sample frequncy of the signal        
        fmin: int/float 
            The minimum frequency
        fmax: int/float 
            The maximum frequency
        powerLaw : float/int (optional)
            This is the power used to describe the frequency evolution. Any positive non
            zero number is acceptable and it will create a power law f=~ (t-tc)**(powerLaw).
            In the case of decimal powers (t-tc) cannot be negative. For this reason we use
            the antisymetric form of the powerlaw when t-tc < 0. For example if powerLaw is 0.5
        tc: float,int (optional)
            This critical time value is the time where the frequency will be fmin. It can
            take values from 0 to duration. If not specified it is set to 0.
        powerSymmetry: {0,1,-1} (optional)
            This value specifies what will happen befor tc. 1 is for symmetrical around tc.
            -1 for antisymmetrical and 0 for random choice. There are cases like when powerLaw 
            is les than 1 that this value can be only -1. It will automaticly set itself if it
            is set wrong. If not specified it is set to symmetry.
        phase: float/int (optional)
            Phase of the signal. If not specified it is set to 0.
        enveloped: bool (optional)
            If set to True it returns the signal within a sigmoid envelope on the edges.
            If not specified it is set to False.
        returnFrequency bool (optional)
            If set to True it will also return the function that describes the frequency.
            
        
        Returns
        ------
        
        numpy.ndarray:
            The chirplet waveform
        
        (numpy.ndarray,numpy.ndarray): in case returnFrequency == True
            The chirplet waveform and the frequency evolution function
            
        Note
        ----
        
        The evolution of the frequency can be tested and experiment with by using the
        returnFrequency=True. This function creates more than the linear chirplet.
    """
    if not (isinstance(fmin,(int,float)) and fmin>=1):
        raise ValueError('fmin must be greater than 1')
    if not (isinstance(fmax,(int,float)) and fmax>fmin):
        raise ValueError('fmax must be greater than fmin')
    if not (isinstance(fs,int) and fs >= 2*fmax):
        raise ValueError('fs must be greater than 2*fmax')
    if not (isinstance(duration,(int,float)) and duration>1/fs):
        raise ValueError('duration must be bigger than 1/fs')
    if powerLaw==None: 
        powerLaw=1
    elif not (isinstance(powerLaw,(int,float)) and powerLaw>0):
        raise ValueError('powerLaw must be a possitive number')
    if powerSymmetry==None:
        powerSymmetry=1
    elif not (powerSymmetry in [0,1,-1]):
        raise ValueError('powerSymmetry must be 0 , 1 or -1,'
                         +' for random selection of symmetry,'
                         +' symmetric or antisymmetric powerLaw respectively')
    if tc==None:
        tc=0
    elif not (isinstance(tc,(int,float)) and 0<=tc<duration):
        raise ValueError('tc must be a nuber in the duration interval')
    if phase == None:
        phase = np.random.rand()*2*np.pi
    elif not (isinstance(phase,(int,float)) and 0<=phase<=2*np.pi):
        raise ValueError('phase must be a number between zero and 2pi')

    
    t=np.arange(0,duration,1/fs)
    
    f=fmin+(fmax-fmin)*norm_decimal_power(t-tc,powerLaw,symmetry=powerSymmetry)


    # FINAL BURST INJECTION
    hp=np.cos(2*np.pi*f*t+phase)   
    hc=np.sin(2*np.pi*f*t+phase)   

    if returnFrequency == True:
        return(hp,hc,f)
    else:
        return (hp,hc)

    

    
    
# def old_chirplet(T,fs
#               ,t0=0
#               ,f0=20+np.random.rand()*30
#               ,fe=50+np.random.rand()*(250)
#               ,phi=np.random.rand()*2*np.pi
#               ,n='default'
#               ,tc='default'
#               ,wnb_envelope=True
#               ,ENV='single'
#               ,envDuration=None
#               ,demo=False):



#     # ENVELOPE FORM

#     ## WNB FORM

#     if wnb_envelope==True:
        
#         wnb_fc = 2+np.random.rand()*10
#         wnb_df = 1+np.random.rand()+5
#         t,wnb=WNB(param=[1, wnb_fc, wnb_df ],T=T,fs=fs,q='noenv')
#     else:
#         t,wnb=np.arange(0,T,1/fs), -0.5*np.ones(fs)


#     ## EDGE SMOOTHING OUTTER ENVELOPE
#     if ENV=='double':
#         q1=0.2+np.random.rand()*0.6
#         q2=0.2+np.random.rand()*0.6
        
#         sig1=2.5+np.random.rand()*2
#         sig2=2.5+np.random.rand()*2

#         a1=0.2+np.random.rand()*0.8
#         a2=0.2+np.random.rand()*0.8

#         e1=a1*envelope(T,fs,q=q1,t0='default',sig=sig1,duration=envDuration)
#         e2=a2*envelope(T,fs,q=q2,t0='default',sig=sig2,duration=envDuration)

#         env=(wnb+1.5)*(e1+e2)

#     elif ENV=='single':
#         q1=0.1+np.random.rand()*0.8
#         env=(wnb+1.5)*envelope(T,fs,q=q1,t0='default',sig=2.5+np.random.rand()*2,duration=envDuration)

#     # FREQUENCY FUNCTION

#     if n=='default':
#         token=np.random.randint(0,2)

#         ## Power 0 < n <1
#         if (-1)**token == -1:
#             n = 0.1+np.random.rand()*0.9
#         ## Power 1 < n < 5
#         elif (-1)**token == 1:
#             n = 1+np.random.rand()*4

#     if tc=='default':
#         tc = 0.2+np.random.rand()*0.6

#     f=f0+((fe-f0)/((1-tc)**n+tc**n))*decimal_power(t-tc,n)+((fe-f0)*(tc**n)/((1-tc)**n+tc**n))



#     # FINAL BURST INJECTION
#     s=env*np.cos(2*np.pi*f*(t-t0)+phi)   

#     if demo==True:
#         fig=plt.figure(figsize=(15,7))

#         gs = gridSpec.GridSpec(2,3, figure=fig)
        
#         ax0=fig.add_subplot(gs[0,0:2])
#         ax0.plot(t,s,'royalblue')
#         ax0.set_title('Waveform Timeseries')
#         ax0.set_xlabel('Time')
#         ax0.set_ylabel('Amplitude')

#         ax0s=fig.add_subplot(gs[1,0:2])
#         ax0s.loglog(np.fft.fftfreq(len(s),1/fs)[0:int(fs/2)],np.abs(np.fft.rfft(s)[:-1]),'darksalmon')
#         ax0s.set_xlim(20,1024)
#         ax0s.set_title('Waveform FFT')
#         ax0s.set_xlabel('Frequency')
#         ax0s.set_ylabel('Amplitude')
        
#         ax1=fig.add_subplot(gs[0,2])
#         ax1.plot(t,f,'blueviolet')
#         ax1.set_title('Frequency change function')
#         ax1.set_ylabel('Frequency')
#         ax1.set_xlabel('Time')

        
#         ax2=fig.add_subplot(gs[1,2])
#         ax2.plot(t,env,'g')
#         ax2.set_title('Envelope function')
#         ax2.set_xlabel('Time')
#         ax2.set_ylabel('Amplitude')        
#         if wnb_envelope==True:
#             ax2.plot(t,wnb+1.5)
#     else:
#         return(t,s)


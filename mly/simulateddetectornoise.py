# function [x, t] = simulateddetectornoise(DET,T,fs,fmin,fmax,seed) 
# % SIMULATEDDETECTORNOISE - Simulate Gaussian colored noise for an IFO.
# % 
# % SIMULATEDDETECTORNOISE generates simulated Gaussian noise with spectrum 
# % matching the design sensitivity curve for a specified gravitational-wave
# % detector, or matching a user-supplied noise curve.
# %  
# %    [n, t] = simulateddetectornoise(DET,T,fs,fmin,fmax)
# %  
# %    DET   String or Vector. String: name of the detector to be simulated.
# %          Must be one of those recognized by the function SRD, or 'white'
# %          (not case sensitive). Vector: one-sided PSD sampled at 
# %          frequencies [1/T:1/T:fs/2] (i.e. all positive frequencies up to
# %          the Nyquist of the desired output data).
# %    T     Scalar. Duration (s) of noise data stream.  T and fs should
# %          satisfy T*fs=integer.
# %    fs    Scalar. Sampling frequency (Hz) of data stream.  T and fs should
# %          satisfy T*fs=integer.
# %    fmin  Scalar. Minimum desired frequency (Hz).
# %    fmax  Scalar. Maximum desired frequency (Hz).
# %    seed  Optional scalar. Seed for randn generator used to produce the
# %          noise data.  See randn.
# %
# %    n     Column vector of length round(fs*T).  Noise timeseries (strain).
# %    t     Column vector of length round(fs*T).  Times (s) at which noise
# %          vector is sampled, starting from t=0.
# %  
# % The design noise spectra are generated from the function SRD.  The noise
# % power spectrum of the simulated data drops as f^(-2) outside the range 
# % [fmin,fmax].  For 'white', the output data is zero-mean unit-variance
# % Gaussian noise and white over the full frequency band.  
# %    
# % For information on the conventions used for discrete Fourier transforms,
# % see FourierTransformConventions.m.  For examples of how to use this
# % function, do "type simulateddetectornoise.m" and see the comments below
# % this help information.
# %
# % original version: 
# %   Patrick J. Sutton, 2005.04.09 <patrick.sutton@astro.cf.ac.uk>
# %  
# %  $Id: simulateddetectornoise.m 5491 2017-07-20 11:12:44Z paul.schale@LIGO.ORG $

# % Note: The Virgo collaboration document VIR-NOT-ROM-1390-090, authors S. Frasca
# % and M. A. Papa, discuss a technique for constructing simulated Gaussian noise
# % based on a spectrum.  Of particular interest is a technique for constructing 
# % lines in the time domain which are then added to the data.  From page 6, the 
# % line contribution is the real part of the process y_i, where
# %
# %    y_i = x_i + w y_{i-1},
# %
# % x is a white complex Gaussian noise process, w is the complex number 
# %
# %    w = exp{i theta} exp{-dt/tau},
# %
# % and dt is the sampling time.  Three input parameters are required:
# % i) the frequency of the line peak, f_0 = theta / (2 pi dt)
# % ii) the bandwidth of the peak, df = 1/(2 pi tau)
# % iii) the peak height of the spectrum, sigma_x^2 = S_{peak} 2 pi (1-|w|^2)/tau.

# % % ---- Test/demonstration code for making simulated data.
# % ifo = {'aLIGO','aVirgo', 'ETB', 'ETC', 'GEO', 'GEO-HF', 'LIGO', ...
# %        'LIGO-LV', 'LHO', 'LLO', 'LISA', 'TAMA', 'Virgo', 'Virgo-LV'};
# % df = 1;
# % fmin = 40;
# % fmax = 2000;
# % f0 = [fmin:df:fmax]';
# % T = 16;
# % fs = 16384;
# % seed = 12345;
# % for ii = 1:length(ifo)
# %     S0 = SRD(ifo{ii},f0);
# %     [x, t] = simulateddetectornoise(ifo{ii},T,fs,fmin,fmax,seed);
# %     [S, F] = medianmeanaveragespectrum(x,fs,fs/df);
# %     figure; set(gca,'fontsize',16)
# %     loglog(F,S.^0.5,'b-')
# %     hold on; grid on
# %     loglog(f0,S0.^0.5,'k-','linewidth',2)
# %     xlabel('frequency (Hz)');
# %     ylabel('strain noise amplitude');
# %     title(ifo{ii})
# %     legend('simulated data','design spectrum',2)
# % end

from .SRD import *
import numpy as np

# There are 3 modes for this function.
# a) Simulated noise from the detector curves without any complexity          DET=SDR string
# b) Simulated noise as a purtubation of a real noise PSD (sudo-real noise)   DET=[freq, PSD]
# c) Simulated noise following a psd that could be anything                   DET=[PSD] size: T*fs/2


def simulateddetectornoise(DET,T,fs,fmin,fmax):#,seed):
    
    #DET: Can be a string vector, a PSD vector of size T*fs/2 or
    #     it can be a size two array with frequencies and values of a PSD (sudodetectornoize)

    #% ---- Checks.
    ###narginchk(5,6);
    if (T>0 and fs>0 and fmin>0 and fmax>0)!=True :
        raise ValueError('Input duration, sampling rate, and minimum and maximum frequencies must be positive.')
    
    if fmax<=fmin :
        raise ValueError('Maximum frequency must be > minimum frequency.')
    
    if fmax>fs/2.:
        raise ValueError('Maximum frequency must be <= Nyquist frequency fs/2.')
    if type(DET)!=str :
        #% ---- If not a string, it must be a vector of noise PSD values at
        #%      frequencies [1/T:1/T:fs/2]. Verify that it is a vector with the
        #%      correct length. 
        if len(DET)==2 and len(DET[0])!=len(DET[1]):
            raise TypeError('Frequency vector and PSD vector are not the same size')
        
        elif np.size(DET)==len(DET) and len(DET)!= T*fs/2:
            print('Size of DET variable: ',str(np.size(DET)))
            print('Expected number of elements: ',str(T*fs/2))
            raise TypeError('The PSD vector does not have the correct size')

    #     if (nargin==6)
    #         if (~isscalar(seed))  % -- isscalar, isvector not in MatLab R13
    #         if (max(size(seed))>1)
    #             error('Seed value must be a scalar.')
    #         else
    #             % ---- Set state of randn. 
    #             randn('state',seed);
    #         end
    #     end
    #% ---- Number of data points, frequency spacing.
    N = np.round(fs*T)
    #% ---- Time at each sample, starting from zero.
    t = np.arange(0,N)/fs
    #% ---- If user has requested white noise, make that and exit.
    if type(DET)==str and DET.lower()=='white':
        x = np.random.randn(N,1)
        return x, t

    #% ---- Make vector of positive frequencies up to Nyquist.
    f = np.arange(1/(1.0*T),fs/2+1/(1.0*T),1/(1.0*T))

    #% ---- Get one-sided SRD (science requirements design) power spectrum (PSD)
    #%      and minimum frequency at which that design is valid. 
    if isinstance(DET,str)==False:
        
        f1,S1=DET[0],DET[1]
        
        PSD_int = interp1d(f1,S1,bounds_error=True)

        fstop = [f1[0],fs/2]
    else:
        PSD_int, fstop = SRD(DET,f)

    #% ---- Print warning if requested frequencies go below fstop.
    if fmin<fstop[0]:
        print('WTruncating data at '+str(fstop[0])+' Hz.')

    #% ---- Print warning if requested frequencies go below fstop.
    if fmax>fstop[1]:
        print('Truncating data at '+str(fstop[1])+' Hz.')

    #% ---- Convert to two-sided power spectrum.
    #PSD = PSD/2 Its not working on python so its merged later

    #% ---- Force noise spectrum to go to zero smoothly outside desired band.

    ###k = find(f<fmin|f<fstop(1))
    k_m=[]
    k_p=[]
    f_w=[] # frequencies that are inside the interpolation range
    for i in range(0,len(f)):
        if f[i]<fmin or f[i]<fstop[0]:
            k_m.append(i)
        elif f[i]>fmax or f[i]>fstop[1]:
            k_p.append(i)
        else:
            f_w.append(f[i])

    f_w=np.array(f_w)

    PSD=PSD_int(f_w)/2

    PSD_frondtail=[]
    if len(k_m)>0:
        for i in range(0,len(k_m)):
            PSD_frondtail.append(PSD[0]*(f[k_m[i]]/f[k_m[-1]])**2)
        PSD=np.hstack((np.array(PSD_frondtail),PSD))

    PSD_backtail=[]
    ##k = find(f>fmax|f>fstop(2));
    if len(k_p)>0:
        for i in range(0,len(k_p)):
            PSD_backtail.append((PSD[-1]*2)/(1+(f[k_p[i]]/f_w[-1])**2))
        PSD=np.hstack((PSD,np.array(PSD_backtail)))



    #% ---- Make white Gaussian random noise in frequency domain, at positive
    #%      frequencies (real and imaginary parts).
    reXp = np.random.randn(len(f))
    imXp = np.random.randn(len(f))
    
    #% ---- Color noise by desired amplitude noise spectrum.
    Xp = ((T*PSD)**0.5)*(reXp + 1j*imXp)/2**0.5; 

    #% ---- Make noise at DC and negative frequencies, pack into vector in usual
    #%      screwy FFT order: 
    #%         vector element:   [ 1  2  ...  N/2-1  N/2    N/2+1            N/2+2   ... N-1  N ]
    #%         frequency (df):   [ 0  1  ...  N/2-2  N/2-1  (N/2 or -N/2)   -N/2+1  ... -2   -1 ]
    #%         F = [ 0:N/2 , -N/2+1:-1 ]'*df
          
    X = np.hstack((np.array([0]),Xp,np.conj(Xp[-2::-1]))) ###??? The last part was conjugated in matlab but here its not

    #% ---- Inverse fft back to time domain, casting off small imaginary part
    #%      (from roundoff error).
    x = np.real(np.fft.ifft(fs*X))

    #% ---- Done
    return 2*PSD,x, t

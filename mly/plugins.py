import matplotlib.pyplot as plt
import numpy as np
from .tools import correlate
class PlugIn:
    
    """PlugIn is a class to encapsulate any additional data we want to
    have along with the main data a DataPod has. The new data we introduce
    with a PlugIn object might be just a value or something that is calcukated
    from other PlugIns or strain already present in the DataPod instance. Every
    PlugIn has a name and a generation function. Additionally it can have
    attributes (related to the attributes use to create the plugin data) and
    also a plot function that will help plot the data from the DataPod.plot(<name>).
    
    When the PlugIn is added to a DataPod through DataPod.addPlugIn() method
    it will create a new attribute for the DataPod to be called like strain 
    DataPod.<name>.
    
    Attributes
    ---------
    
    name: str
        The name you want to use for the data you add. This is going to be an
        attribute of the DataPod object that the plugin will be added.
        
    genFunction: function/value
        The fuction to use to infere the new data from other attributes in the 
        targeted DataPod. You can also just pass a value if no use of other 
        attributes is needed.

    plotFunction: function (optional)
        A function with the same attributes as genFunction that returns a
        matplotlib.pyplot plot. This will be called if you want to plot 
        the data that the plugin will have.
        
    attributes: list/tuple of strings 
        A list or tuple of the names of attributes used by the genFunction.
        These attributes must be attributes of the DataPod you plan to add 
        the plugin.
        
    plotAttributes: list/tuple of strings 
        A list or tuple of the names of attributes used by the plotFunction.
        These attributes must be attributes of the DataPod you plan to add 
        the plugin. If not specified it is set equal to attributes.
                    
            
    Notes
    -----
    
    When you call plotFunction, make sure that genFunction and plotFunction
    call the same attributes with the same order. 
    """
    
    def __init__(self
                 ,name
                 ,genFunction
                 ,plotFunction=None
                 ,attributes=None
                 ,plotAttributes=None
                 ,**kwargs):

        self.name = name
        self.genFunction=genFunction
        self.plotFunction=plotFunction
        self.attributes=attributes
        self.plotAttributes=plotAttributes
        self.kwargs=kwargs

    
    @property
    def name(self):
        return self._name                 
    @name.setter
    def name(self,_name):
        if isinstance(_name,str) and not any(ic in _name for ic in " -!@£$%^&*()±§?\"|><.,`~+=:;'" ):
            self._name=_name
        else:
            raise AttributeError("Name must be a string with valid characters")
            
    @property
    def genFunction(self):
        return self._genFunction               
    @genFunction.setter
    def genFunction(self,function):
        self._genFunction=function

#         if callable(function):
#             self._genFunction=function
#         else:
#             def token():
#                 return function
#             self._genFunction=token

    @property
    def plotFunction(self):
        return self._plotFunction               
    @plotFunction.setter
    def plotFunction(self,function):
        if function==None:
            self._plotFunction=None
        elif callable(function):
            self._plotFunction=function
        else:
            raise TypeError('plotFunction must be a callable object')
            
    @property
    def attributes(self):
        return self._attributes                 
    @attributes.setter
    def attributes(self,attributes):
        if attributes==None:
            attributes =[]
        if not isinstance(attributes,(list,tuple)):
            raise TypeError("Attributes must be in a list or tuple")
        if not all(isinstance(at,str) for at in attributes):
            raise TypeError("Attributes can only be strings")
        self._attributes=attributes  
        
    @property
    def plotAttributes(self):
        return self._plotAttributes                 
    @plotAttributes.setter
    def plotAttributes(self,plotattributes):
        if plotattributes==None:
            if self.attributes==None:
                plotattributes =[]
            else:
                plotattributes=self.attributes
        if not isinstance(plotattributes,(list,tuple)):
            raise TypeError("plotAttributes must be in a list or tuple")
        if not all(isinstance(at,str) for at in plotattributes):
            raise TypeError("Attributes can only be strings")
        self._plotAttributes=plotattributes
    

# Default PlugIn objects

known_plug_ins=['snr','hrss','psd','correlation','correlation_12','correlation_30']

def correlationFunction(strain,detectors,fs,window=None):
    if window==None:
        window=int(0.043*fs)
    correlations=[]
    for i in range(len(detectors)):
        for j in range(1+i,len(detectors)):
            correlations.append(correlate(strain[i],strain[j],window).tolist())
    return(np.array(correlations))
    
def plotcorrerlaion(strain,detectors,fs,data=None):
    
    f,ax = plt.subplots(figsize=(15,7))#,facecolor='lightslategray')
    ax.set_xlabel('Time Shift')
    ax.set_ylabel('Pearson Correlation')
    tlength=len(data[0])/fs
    tarray=np.arange(-tlength/2,tlength/2,1/fs)
    count_=0
    for i in np.arange(len(detectors)):
        for j in np.arange(i+1,len(detectors)):
            ax.plot(tarray,data[count_]
                     ,label=str(detectors[i])+str(detectors[j]))
            plt.legend()
            count_+=1
    return ax


# correlationPlugIn =  PlugIn(name='correlation'
#                       ,genFunction=correlationFunction
#                       ,attributes=['strain','detectors','fs']
#                       ,plotFunction=plotcorrerlaion)



def knownPlugIns(name,**kwargs):
    if name not in known_plug_ins:
        raise TypeError("Name must be a string and one of known plugins "+str(known_plug_ins))
    if name=='correlation':
        if 'window' in list(kwargs.keys()):
            w=kwargs['window']
        else:
            w=None
        plugin=PlugIn(name='correlation'
                      ,genFunction=correlationFunction
                      ,attributes=['strain','detectors','fs']
                      ,plotAttributes=['strain','detectors','fs']
                      ,plotFunction=plotcorrerlaion
                      ,window=w)
        
    elif 'correlation'==name.split('_')[0] and isinstance(int(name.split('_')[-1]),int):
        
        w=int(name.split('_')[-1])
        plugin=PlugIn(name='correlation'
                      ,genFunction=correlationFunction
                      ,attributes=['strain','detectors','fs']
                      ,plotAttributes=['strain','detectors','fs']
                      ,plotFunction=plotcorrerlaion
                      ,window=w)
    else:
        raise ValueError(name+ 'is not a known PlugIn object')
    
    return plugin
        



def plotpsd(detectors,fs,data=None):
    f,ax = plt.subplots(figsize=(15,7))
    
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power Spectral Density')
    colors = {'H': '#ee0000','L':'#4ba6ff','V':'#9b59b6','K':'#ffb200','I':'#b0dd8b','U': 'black'}
    f=np.arange(0,int(fs/2)+1)

    for det in detectors:
        ax.loglog(f,data[detectors.index(det)],color= colors[det]
                   ,label = str(det))
    plt.legend()
    
    return ax

    
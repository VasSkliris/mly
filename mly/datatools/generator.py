def generator(duration
               ,fs
               ,size
               ,detectors
               ,injectionFolder = None
               ,labels = None
               ,backgroundType = None
               ,injectionSNR = None
               ,noiseSourceFile = None
               ,windowSize = None #(32)            
               ,timeSlides = None #(1)
               ,startingPoint= None #(32)
               ,name = None
               ,savePath = None
               ,single = False  # Making single detector injections as glitch
               ,injectionCrop = 0  # Allows to crop part of the injection when you move the injection arroud, 0 is no 1 is maximum means 100% cropping allowed. The cropping will be a random displacement from zero to parto of duration
               ,frames=None 
               ,channels=None
               ,disposition=None
               ,maxDuration=None
               ,differentSignals=False   # In case we want to put different injection to every detector.
               ,plugins=None
               ,**kwargs):

    """This is one of the core methods/function of this module.
    It generates data instances of detector strain data that could be 
    from real detectors or simulated noise. It has many options for
    each different type of instance -that it can generate.

    Parameters
    ----------

    duration : float/int
        The duration of instances to be generated in seconds.

    fs : float/ing
        The sample frequency of the instances to be generated.

    size : int
        The number of instances to be generated.

    detectors : str/ list of str
        The detectors to be used for the background. Even for optimal 
        gaussian noise the background is different for each detector. 
        For example if you wan to include Hanford, Livingston and Virgo
        you state it as 'HLV' or ['H','L','V'].

    injectionFolder: str (path, optional)
        The path to the directory with the injections. If not specified
        the generator will just generate noise instances, setting
        injectionSNR to 0.

    labels: dict (optional)
        The optional label for all the instances to be generated. 
        If not specified labels is {'type':'UNDEFINED'}.

    backgroundType: {'optimal','real'}
        The type of background to be used. If 'optimal' is chosen,
        a simulated noise background will be used (simulateddetectornoise.py)
        depending the detector. If 'real' is chosen, a source of real detector
        noise will need to be specified from noiseSourceFile.

    injectionSNR: float/int
        The Signal to Noise Ration of the injection used for the instances.
        All injections are calibrated to be in that SNR value.

    noiseSourceFile: {gps intervals, path, txt file, numpy.ndarray}
        The source of real noise. There are many options to provide source 
        of real noise. 

        * A list intervals (one for each detectors) with gps times. The script 
        will get the available coinsident data within the 'detectors' and use them.

        * A path with txt or DataPod files that have noise data.

        * A numpy.ndarray with the data directly.

        This parameter evolves depending on the needs of users.

    windowSize: int/float
        The amount of seconds to use around the instance duration, for the 
        calculation of PSD. The bigger the windowSize the better whittening.
        Although the calculation takes more time. It defaults to 16 times
        the duration parameter.

    timeSlides: int (optional)
        The number of timeshifts to use on the provided noise source.
        If noise source has N instances of utilisable noise by using
        timeSlides we will have N*(1-timeSlides) instances. Default
        is 1 which means not using timeSlides.

    startingPoint: float/int (optional)
        The starting point in seconds from which we will use the noise
        source. In case we have a big noise sample by specifing startingPoint
        we do not start from the beggining but from that second. Default is 0.

    name : str (optional)
        The name of the dataset, if it is saved. 

    savePath : str (path, optional)
        The of the file to be saved. If not stated file will not be saved and
        it will just be returned.

    single : bool
        If true and injectionSNR or injectionHRSS is defined, it will add
        a signal to only one of the available detectors, randomly selected.
        All SNR or hrss will be used on the single injection. Default is False.

    differentSignals : bool
        If true each detector gets a different randomly selected signal. 
        Default is False.

    injectionCrop : float [0,1]
        Allows to crop part of the injection (up to the value provided when 
        randomly positions the injection arroud on the background. 0 means
        no croppin allowd and 1 means maximum 100% cropping allowed. The cropping 
        will be a random displacement from zero to parto of durationn.

    frames: dict or {C00, C01,C02}
        In case we use real detector noise and the script looks for the available 
        noise segments we need to specify the frames to look for. We have to provide 
        a dictionary of the frames for the given detectors. C00, C01 and C02 provides a
        shortcut for the widly used frames. 
        {'H': 'H1_HOFT_C0X' ,'L': 'L1_HOFT_C0X' ,'V': 'V1Online'}.
        C02 is chossen as default given that is pressent in all observing runs.

    channels: dict or {C00, C01,C02} 
        As with frames, in case we use real detector noise and the script looks for 
        the available noise segments we need to specify the channels to look for. 
        We have to provide a dictionary of the channels for the given detectors. 
        C00, C01 and C02 provides a shortcut for the widly used frames. 
        {'H': 'H1:DCS-CALIB_STRAIN_C0X','L': 'L1:DCS-CALIB_STRAIN_C0X' ,'V': 'V1:Hrec_hoft_16384Hz'}.
        C02 is chossen as default given that is pressent in all observing runs.

    disposition: float (optional)
        Disposition is the time interval in seconds within we want the central times of the 
        signals to appear. It is used with differentSignals = True. The bigger
        the interval the bigger the spread. Although the spread is always constrained by
        the duration of the longest signa selected so that the signals don't get cropped.

    maxDuration : float (optional)
        If you want to use injections and want to restrict the maximum duration used.
        This will check the randomly selected injection's duration. If it is bigger
        that the one specified here, it will select another random injection. This will
        continue until it finds one with duration smaller than maxDuration parameter.
        Note that the smaller this parameter is the more time it will take to complete
        the generation.

    plugins : list of strings
        You can specify here plugins that are included in plugins.known_plug_ins variable.
        Those plugins will automaticly be added into the dataset returned.


    Returns
    -------

    A DataSet object. If savePath is defined, it saves the DataSet to that path
    and it returns None.



    """
    # Integration limits for the calculation of analytical SNR
    # These values are very important for the calculation

    fl, fm=20, int(fs/2)#

    profile = {'H' :'aligo','L':'aligo','V':'avirgo','K':'KAGRA_Early','I':'aligo'}

    # Labels used in saving file
    #lab={10:'X', 100:'C', 1000:'M', 10000:'XM',100000:'CM'}  

    lab={}
    if size not in lab:
        lab[size]=str(size)

    # ---------------------------------------------------------------------------------------- #   
    # --- duration --------------------------------------------------------------------------- #

    if not (isinstance(duration,(float,int)) and duration>0 ):
        raise ValueError('The duration value has to be a possitive float'
            +' or integer representing seconds.')

    # ---------------------------------------------------------------------------------------- #   
    # --- fs - sample frequency -------------------------------------------------------------- #

    if not (isinstance(fs,int) and fs>0):
        raise ValueError('Sample frequency has to be a positive integer.')

    # ---------------------------------------------------------------------------------------- #   
    # --- detectors -------------------------------------------------------------------------- #

    if isinstance(detectors,(str,list)):
        for d in detectors:
            if d not in ['H','L','V','K','I']: 
                raise ValueError("detectors have to be a list of strings or a string"+
            " with at least one the followings as elements: \n"+
            "'H' for LIGO Hanford \n'L' for LIGO Livingston\n"+
            "'V' for Virgo \n'K' for KAGRA \n'I' for LIGO India (INDIGO) \n"+
            "\n'U' if you don't want to specify detector")
        if isinstance(detectors,str): detectors=list(detectors)
    else:
        raise ValueError("detectors have to be a list of strings or a string"+
            " with at least one the followings as elements: \n"+
            "'H' for LIGO Hanford \n'L' for LIGO Livingston\n"+
            "'V' for Virgo \n'K' for KAGRA \n'I' for LIGO India (INDIGO) \n"+
            "\n'U' if you don't want to specify detector")


    # ---------------------------------------------------------------------------------------- #   
    # --- size ------------------------------------------------------------------------------- #        

    if not (isinstance(size, int) and size > 0):
        raise ValueError("size must be a possitive integer.")

    # ---------------------------------------------------------------------------------------- #   
    # --- injectionFolder -------------------------------------------------------------------- #

    if injectionFolder == None:
        pass

    # Path input
    elif isinstance(injectionFolder, str): 

        # directory with pods path
        if os.path.isdir(injectionFolder):

            if injectionFolder[-1] != "/": injectionFolder+="/" 

            if all(os.path.isdir(injectionFolder+det) for det in detectors):

                inj_type = 'oldtxt'
            else:
                inj_type = 'directory'

        # dataPod or dataSet path
        elif (os.path.isfile(injectionFolder) and injectionFolder[-4:]=='.pkl'):
            with open(injectionFolder,'rb') as obj:
                injectionFolder = pickle.load(obj)
            if isinstance(injectionFolder,DataPod):
                inj_type = 'DataPod'
            elif isinstance(injectionFolder,DataSet):
                if len(injectionFolder) > 0:
                    inj_type = 'DataSet'
                else:
                    raise ValueError("injectionFolder DataSet is empty")

        else:
            raise FileNotFoundError('Not valid directory for :'+injectionFolder)
    # DataSet                     
    elif isinstance(injectionFolder,DataSet):

        if len(injectionFolder) > 0:
            inj_type = 'DataSet'
        else:
            raise ValueError("injectionFolder DataSet is empty")

    # DataPod
    elif isinstance(injectionFolder,DataPod):
        inj_type = 'DataPod'

    else:
        raise Type('Not valid input for injectionFolder:'+injectionFolder
                    ,"\nIt has to be either a folder or a DataPod or DataSet object.") 

    # ---------------------------------------------------------------------------------------- #   
    # --- labels ----------------------------------------------------------------------------- #

    if labels == None:
        labels = {'type' : 'UNDEFINED'}
    elif not isinstance(labels,dict):
        raise TypeError(" Labels must be a dictionary.Suggested keys for labels"
                        +"are the following: \n{ 'type' : ('noise' , 'cbc' , 'signal'"
                        +" , 'burst' , 'glitch', ...),\n'snr'  : ( any int or float number "
                        +"bigger than zero),\n'delta': ( Declination for sky localisation,"
                        +" float [-pi/2,pi/2])\n'ra'   : ( Right ascention for sky "
                        +"localisation float [0,2pi) )})")

    # ---------------------------------------------------------------------------------------- #   
    # --- backgroundType --------------------------------------------------------------------- #

    if backgroundType == None:
        backgroundType = 'optimal'
    elif not (isinstance(backgroundType,str) 
          and (backgroundType in ['optimal','sudo_real','real'])):
        raise ValueError("backgroundType is a string that can take values : "
                        +"'optimal' | 'sudo_real' | 'real'.")

    # ---------------------------------------------------------------------------------------- #   
    # --- injectionSNR ----------------------------------------------------------------------- #
    if injectionFolder == None :
        injectionSNR = 0
#         elif (injectionFolder != None and injectionSNR == None ):
#             raise ValueError("If you want to use an injection for generation of"+
#                              "data, you have to specify the SNR you want.")
#         elif injectionFolder != None and (not (isinstance(injectionSNR,(int,float)) 
#                                           and injectionSNR >= 0)):
#             raise ValueError("injectionSNR has to be a positive number")


    # ---------------------------------------------------------------------------------------- #   
    # --- noiseSourceFile -------------------------------------------------------------------- #

    if (backgroundType == 'sudo_real' or backgroundType =='real'):
        # if noiseSourceFile == None:
        #     raise TypeError('If you use sudo_real or real noise you need'
        #         +' a real noise file as a source.')

        # Loading noise using gwdatafind and gwpy
        if (isinstance(noiseSourceFile,list) 
                and len(noiseSourceFile)==len(detectors) 
                and all(len(el)==2 for el in noiseSourceFile)):

            noiseFormat='gwdatafind'

        # Loading noise using file paths of txt files (different for each detector)
        elif (isinstance(noiseSourceFile,list) 
              and len(noiseSourceFile)==len(detectors) 
              and all(isisntance(path_,str) for path_ in noiseSourceFile)
              and all(path_[-4:]=='.txt' for path_ in noiseSourceFile)):

            noiseFormat='txtD'

        # Loading noise using file paths of txt files (one with all detectors)
        elif (isinstance(noiseSourceFile,str) 
              and noiseSourceFile[-4:]=='.txt'):

            noiseFormat='txt1'

        # Loading noise using file paths of txt files (one with all detectors)
        elif (isinstance(noiseSourceFile,np.ndarray) and len(detectors) in noiseSourceFile.shape):

            noiseFormat='array'

        # Loding noise using DataPods or DataSets (one with all detectors)
        elif ((isinstance(noiseSourceFile,str) 
              and noiseSourceFile[-4:]=='.pkl') 
              or (('Pod' in str(type(noiseSourceFile))) or ('Set' in str(type(noiseSourceFile))))):

            noiseFormat='PodorSet'

        else:
            raise TypeError("The noise type format given is not one valid")



    # ---------------------------------------------------------------------------------------- #   
    # --- windowSize --(for PSD)-------------------------------------------------------------- #        

    if windowSize == None: windowSize = duration*16
    if not isinstance(windowSize,int):
        raise ValueError('windowSize needs to be a number')
    if windowSize < duration :
        raise ValueError('windowSize needs to be bigger than the duration')

    # ---------------------------------------------------------------------------------------- #   
    # --- timeSlides ------------------------------------------------------------------------- #

    # if timeSlides == None: timeSlides = 0
    # if not (isinstance(timeSlides, int) and timeSlides >=0) :
    #     raise ValueError('timeSlides has to be an integer equal or bigger than 0')

    # ---------------------------------------------------------------------------------------- #   
    # --- startingPoint ---------------------------------------------------------------------- #

    if startingPoint == None : startingPoint = 0 
    if not (isinstance(startingPoint, (float,int)) and (startingPoint%duration)%(1/fs) ==0) :
        raise ValueError('Starting point decimal part must always be a multiple of time step')        

    # ---------------------------------------------------------------------------------------- #   
    # --- name ------------------------------------------------------------------------------- #

    if name == None : name = ''
    if not isinstance(name,str): 
        raise ValueError('name optional value has to be a string')

    # ---------------------------------------------------------------------------------------- #   
    # --- savePath --------------------------------------------------------------------------- #

    if savePath == None : 
        pass
    elif (savePath,str): 
        if not os.path.isdir(savePath) : 
            raise FileNotFoundError('No such file or directory:' +savePath)
    else:
        raise TypeError("Destination Path has to be a string valid path")

    # ---------------------------------------------------------------------------------------- #   
    # --- plugins ---------------------------------------------------------------------------- #

    if plugins == None:
        plugins = []
    elif not isinstance(plugins,list):
        plugins = [plugins]
    if isinstance(plugins,list):
        for pl in plugins:
            if pl in known_plug_ins:
                pass
            elif isinstance(pl,str) and ('knownPlugIns' in pl):
                pass
            else:
                raise TypeError("plugins must be a list of PlugIn object or from "+str(known_plug_ins))

    # --- fames ---

    if frames==None or (isinstance(frames,str) and frames.upper()=='C02'):
        frames = {'H': 'H1_HOFT_C02'
                 ,'L': 'L1_HOFT_C02'
                 ,'V': 'V1Online'}
    elif (isinstance(frames,str) and frames.upper()=='C01'):
        frames = {'H': 'H1_HOFT_C01'
                 ,'L': 'L1_HOFT_C01'
                 ,'V': 'V1Online'}
    elif (isinstance(frames,str) and frames.upper()=='C00'):
        frames = {'H': 'H1_HOFT_C00'
                 ,'L': 'L1_HOFT_C00'
                 ,'V': 'V1Online'}
    elif not (isinstance(frames,dict)): 

          raise ValueError("Frame type "+str(frames)+" is not valid")

    if channels==None or (isinstance(channels,str) and channels.upper()=='C02'):
        channels = {'H': 'H1:DCS-CALIB_STRAIN_C02'
                   ,'L': 'L1:DCS-CALIB_STRAIN_C02'
                   ,'V': 'V1:Hrec_hoft_16384Hz'}
    elif (isinstance(channels,str) and channels.upper()=='C01'):
        channels = {'H': 'H1:DCS-CALIB_STRAIN_C01'
                   ,'L': 'L1:DCS-CALIB_STRAIN_C01'
                   ,'V': 'V1:Hrec_hoft_16384Hz'}
    elif (isinstance(channels,str) and channels.upper()=='C00'):
        channels = {'H': 'H1:GDS-CALIB_STRAIN'
                   ,'L': 'L1:GDS-CALIB_STRAIN'
                   ,'V': 'V1:Hrec_hoft_16384Hz'}
    elif not (isinstance(channels,dict)): 

          raise ValueError("Channel type "+str(channels)+" is not valid")

    #print(frames,channels)


    # --- disposition


    # If you want no shiftings disposition will be zero
    if disposition == None: disposition=0
    # If you want shiftings disposition has to adjust the maximum length of an injection
    if isinstance(disposition,(int,float)) and 0<=disposition<duration:
        disposition_=disposition
        maxDuration_ = duration-disposition_
    # If you want random shiftings you can define a tuple or list of those random2*[[eventgps-windowSize/2,eventgps+windowSize/2]] shiftings.
    # This will adjust also
    elif (isinstance(disposition,(list,tuple))):
        if not(len(disposition)==2 
            and isinstance(disposition[0],(int,float))
            and isinstance(disposition[1],(int,float)) 
            and 0<=disposition[0]<duration 
            and disposition[0]<disposition[1]<duration):

            raise ValueError('If you want random range of dispositons '
                            +'it needs to be a tuple or list of two numbers that represent a range')
        else: disposition_=disposition[1]
    else:
        raise TypeError('Disposition can be a number or a range'
                        +' of two nubers (start,end) that always is less than duration')

    # --- max Duration

    if maxDuration == None: 
        maxDuration=duration
    if not (isinstance(maxDuration,(int,float)) and 0<maxDuration<=duration):
        raise ValueError('maxDuration must be between 0 and duration')
    else:
        maxDuration_=min(duration-disposition_,maxDuration)

    # Making a list of the injection names,
    # so that we can sample randomly from them

    injectionFileDict={}
    noise_segDict={}
    for det in detectors:

        if injectionFolder == None:
            injectionFileDict[det] = None
        elif inj_type == 'oldtxt':
            injectionFileDict[det] = dirlist(injectionFolder+'/' + det)

        elif inj_type == 'directory':
            injectionFileDict[det] = dirlist(injectionFolder)

        elif inj_type == 'DataPod':
            injectionFileDict[det] = [injectionFolder]

        elif inj_type == 'DataSet':
            injectionFileDict[det] = [injectionFolder.dataPods]

        else:
            raise TypeError("Unknown type of injections")



    # --- PSDm PSDc
    if 'PSDm' in kwargs: 
        PSDm=kwargs['PSDm']
    else:
        PSDm=None

    if 'PSDc' in kwargs: 
        PSDc=kwargs['PSDc']
    else:
        PSDc=None

    if PSDm==None: 
        PSDm={}
        for det in detectors:
            PSDm[det]=1
    if PSDc==None: 
        PSDc={}
        for det in detectors:
            PSDc[det]=0
    # ------------------------------
    # --- injectionHRSS ------------

    if 'injectionHRSS' in kwargs:
        injectionHRSS = kwargs['injectionHRSS']
    else: 
        injectionHRSS=None

    if injectionFolder == None :
        injectionHRSS = None
    # ------------------------------
    if 'ignoreDetector' in kwargs: 
        ignoreDetector=kwargs['ignoreDetector']
    else:
        ignoreDetector=None


    if backgroundType == 'optimal':
        magic={1024: 2**(-21./16.), 2048: 2**(-23./16.), 4096: 2**(-25./16.), 8192: 2**(-27./16.)}
        param = magic[fs]

    elif backgroundType in ['sudo_real','real']:
        param = 1
        gps0 = {}
        if noiseFormat=='txtD':

            for det in detectors:
                noise_segDict[det] = np.loadtxt(noiseSourceFile[detectors.index(det)])

                gps0[det]=float(noiseSourceFile[detectors.index(det)].split('_')[1])

            ind=internalLags(detectors = detectors
                               ,lags = timeSlides
                               ,duration = duration
                               ,fs = fs
                               ,size = int(len(noise_segDict[detectors[0]])/fs
                                           -startingPoint-(windowSize-duration))
                               ,start_from_sec=startingPoint)

        elif noiseFormat=='txt1':

            file_=np.loadtxt(noiseSourceFile)

            if len(detectors) not in file_.shape: 
                raise ValueError(".txt file provided for noise doesn't have equal "
                                 +"to detector number entries of noise")

            for det in detectors:
                noise_segDict[det] = file_[detectors.index(det)]

                gps0[det]=float(noiseSourceFile.split('_')[1])

            ind=internalLags(detectors = detectors
                               ,lags = timeSlides
                               ,duration = duration
                               ,fs = fs
                               ,size = int(len(noise_segDict[detectors[0]])/fs
                                           -startingPoint-(windowSize-duration))
                               ,start_from_sec=startingPoint)

        elif noiseFormat=='array':

            file_=noiseSourceFile

            if len(detectors) not in file_.shape: 
                raise ValueError(".txt file provided for noise doesn't have equal "
                                 +"to detector number entries of noise")

            for det in detectors:
                noise_segDict[det] = file_[detectors.index(det)]

                gps0[det]=0

            ind=internalLags(detectors = detectors
                               ,lags = timeSlides
                               ,duration = duration
                               ,fs = fs
                               ,size = int(len(noise_segDict[detectors[0]])/fs
                                           -startingPoint-(windowSize-duration))
                               ,start_from_sec=startingPoint)

        elif noiseFormat=='PodorSet':

            if isinstance(noiseSourceFile,str):
                with open(noiseSourceFile,'rb') as obj:
                    file_ = pickle.load(obj)
            else:
                file_=noiseSourceFile

            if 'Pod' in str(type(file_)):
                noiseFormat = 'DataPod'
                for det in detectors:
                    noise_segDict[det] = file_.strain[detectors.index(det)]

                    gps0[det]=float(file_.gps[detectors.index(det)])


                ind=internalLags(detectors = detectors
                                   ,lags = timeSlides
                                   ,duration = duration
                                   ,fs = fs
                                   ,size = int(len(noise_segDict[detectors[0]])/fs
                                               -startingPoint-(windowSize-duration))
                                   ,start_from_sec=startingPoint)

                if size > int(len(noise_segDict[detectors[0]])/fs
                                               -startingPoint-(windowSize-duration)):
                    print("Requested size is bigger that the noise sourse data"
                                     +" can provide. Background will be used multiple times")

                    indexRepetition = ceil(size/int(len(noise_segDict[detectors[0]])/fs
                                               -startingPoint-(windowSize-duration)))

                    for det in detectors:
                        ind[det] = indexRepetition*ind[det]

            elif 'Set' in str(type(file_)):
                noiseFormat = 'DataSet'
                for podi in range(len(file_)):

                    if podi==0:
                        for det in detectors:
                            noise_segDict[det] = [file_[podi].strain[detectors.index(det)]]

                            gps0[det]=float(file_[podi].gps[detectors.index(det)])
                            if len(file_[podi].strain[detectors.index(det)])!=windowSize*fs:
                                raise ValueError("Noise source data are not in the shape expected.")

                    else:
                        for det in detectors:
                            noise_segDict[det].append(file_[podi].strain[detectors.index(det)])



                ind=internalLags(detectors = detectors
                                   ,lags = timeSlides
                                   ,duration = duration
                                   ,fs = 1
                                   ,size = len(file_)
                                   ,start_from_sec=startingPoint)
                for det in detectors:
                    print("det",gps0[det]+np.array(ind[det])+(windowSize-duration)/2)

                if size > len(file_):
                    print("Requested size is bigger that the noise sourse data"
                                     +" can provide. Background will be used multiple times")

                    indexRepetition = ceil(size/len(file_))

                    for det in detectors:
                        ind[det] = indexRepetition*ind[det]
                        noise_segDict[det] = indexRepetition*noise_segDict[det]




        elif noiseFormat=='gwdatafind':
            for d in range(len(detectors)):

                for trial in range(1):
                    try:
                        t0=time.time()
                        conn=gwdatafind.connect()
                        urls=conn.find_urls(detectors[d]
                                           , frames[detectors[d]]
                                           , noiseSourceFile[d][0]
                                           , noiseSourceFile[d][1])

                        noise_segDict[detectors[d]]=TimeSeries.read(urls
                                                                    , channels[detectors[d]]
                                                                    , start =noiseSourceFile[d][0]
                                                                    , end =noiseSourceFile[d][1]
                                                                   ).resample(fs).astype('float64').value#[fs:-fs].value
                        # Added [fs:fs] because there was and edge effect

                        print("\n time to get "+detectors[d]+" data : "+str(time.time()-t0))

                        if (len(np.where(noise_segDict[detectors[d]]==0.0)[0])
                            ==len(noise_segDict[detectors[d]])):
                            raise ValueError("Detector "+detectors[d]+" is full of zeros")
                        elif len(np.where(noise_segDict[detectors[d]]==0.0)[0])!=0:
                            print("WARNING : "+str(
                                len(np.where(noise_segDict[detectors[d]]==0.0)[0]))
                                  +" zeros were replased with the average of the array")
                        print("Success on getting the "+str(detectors[d])+" data.")
                        break

                    except Exception as e:
                        print(e.__class__,e)
                        print("/n")
                        print("Failed getting the "+str(detectors[d])+" data.\n")

                        #waiting=140+120*np.random.rand()
                        #os.system("sleep "+str(waiting))
                        #print("waiting "+str(waiting)+"s")
                        continue

                gps0[detectors[d]] = float(noiseSourceFile[d][0])

            ind=internalLags(detectors = detectors
                               ,lags = timeSlides
                               ,duration = duration
                               ,fs = fs
                               ,size = int(len(noise_segDict[detectors[0]])/fs-(windowSize-duration))
                               ,start_from_sec=startingPoint)

#             print(len(ind['H']),len(ind['L']))
#             print(len(noise_segDict['H'])/fs,len(noise_segDict['L'])/fs)



    thetime = time.time()
    DATA=DataSet(name = name)
    for I in range(size):


        t0=time.time()

        detKeys = list(injectionFileDict.keys())

        if single == True: luckyDet = np.random.choice(detKeys)
        if isinstance(disposition,(list,tuple)):
            disposition_=disposition[0]+np.random.rand()*(disposition[1]-disposition[0])
            maxDuration_=min(duration-disposition_,maxDuration)

        index_selection={}
        if injectionFolder != None:
            if differentSignals==True:
                if maxDuration_ != duration:
                    for det in detectors: 
                        index_sample=np.random.randint(0,
                                                  len(injectionFileDict[detKeys[0]]))

                        if inj_type =='oldtxt':
                            sampling_strain=np.loadtxt(injectionFolder+'/'
                                                       +det+'/'+injectionFileDict[det][index_sample])

                        elif inj_type == 'directory':
                            s_pod=DataPod.load(injectionFolder+
                                                         '/'+injectionFileDict[det][index_sample])
                            sampling_strain=s_pod.strain[s_pod.detectors.index(det)]

                        # case where we pass one pod only
                        elif inj_type == 'DataPod':
                            raise TypeError("You cannot have single pod for inejctions and "
                                            +" also differentSignals = True")

                        elif inj_type == 'DataSet':
                            s_pod = injectionFileDict[det][index_sample] 
                            sampling_strain=s_pod.strain[s_pod.detectors.index(det)]

                        index_selection[det]=index_sample

                        while (len(sampling_strain)/fs > maxDuration_ 
                               or list(index_selection.values()).count(index_sample) > 1 ):

                            index_sample=np.random.randint(0,
                                                      len(injectionFileDict[detKeys[0]]))

                            if inj_type=='oldtxt':
                                sampling_strain=np.loadtxt(injectionFolder+'/'
                                                       +det+'/'+injectionFileDict[det][index_sample])

                            elif inj_type == 'directory':
                                s_pod=DataPod.load(injectionFolder+
                                                         '/'+injectionFileDict[det][index_sample])
                                sampling_strain=s_pod.strain[s_pod.detectors.index(det)]

                            elif inj_type == 'DataSet':
                                s_pod = injectionFileDict[det][index_sample] 
                                sampling_strain=s_pod.strain[s_pod.detectors.index(det)]

                            index_selection[det]=index_sample

                else:
                    for det in detectors: 
                        index_sample=np.random.randint(0,
                                                  len(injectionFileDict[detKeys[0]]))

                        index_selection[det]=index_sample

                        while (list(index_selection.values()).count(index_sample) > 1):

                            index_sample=np.random.randint(0,
                                                      len(injectionFileDict[detKeys[0]]))

                            index_selection[det]=index_sample
            else:
                if maxDuration_ != duration:
                    index_sample=np.random.randint(0,len(injectionFileDict[detKeys[0]]))

                    if inj_type =='oldtxt':
                        sampling_strain=np.loadtxt(injectionFolder+'/'
                                                   +det+'/'+injectionFileDict[det][index_sample])

                    elif inj_type == 'directory':
                        s_pod=DataPod.load(injectionFolder+
                                                     '/'+injectionFileDict[det][index_sample])
                        sampling_strain=s_pod.strain[s_pod.detectors.index(det)]

                    elif inj_type == 'DataPod':
                        s_pod=injectionFolder
                        sampling_strain=s_pod.strain[s_pod.detectors.index(det)]         

                    elif inj_type == 'DataSet':
                        s_pod = injectionFileDict[det][index_sample] 
                        sampling_strain=s_pod.strain[s_pod.detectors.index(det)]

                    while (len(sampling_strain)/fs > maxDuration_
                           or index_sample in list(index_selection.values())):

                        index_sample=np.random.randint(0,len(injectionFileDict[detKeys[0]]))

                        if inj_type =='oldtxt':
                            sampling_strain=np.loadtxt(injectionFolder+'/'
                                                       +det+'/'+injectionFileDict[det][index_sample])

                        elif inj_type == 'directory':
                            s_pod=DataPod.load(injectionFolder+
                                                         '/'+injectionFileDict[det][index_sample])
                            sampling_strain=s_pod.strain[s_pod.detectors.index(det)]

                        elif inj_type == 'DataPod':
                            s_pod=injectionFolder
                            sampling_strain=s_pod.strain[s_pod.detectors.index(det)]         

                        elif inj_type == 'DataSet':
                            s_pod = injectionFileDict[det][index_sample] 
                            sampling_strain=s_pod.strain[s_pod.detectors.index(det)]

                    for det in detectors: index_selection[det]=index_sample
                else:
                    index_sample=np.random.randint(0,len(injectionFileDict[detKeys[0]]))
                    for det in detectors: index_selection[det]=index_sample

#                 inj_ind = np.random.randint(0,len(injectionFileDict[detKeys[0]]))

        SNR0_dict={}
        back_dict={}
        inj_fft_0_dict={}
        asd_dict={}
        PSD_dict={}
        gps_list = []



        for det in detKeys:

            # In case we want to put different injection to every detectors.
#                 if differentSignals==True:
#                     inj_ind = np.random.randint(0,len(injectionFileDict[detKeys[0]]))

            if backgroundType == 'optimal':

                # Creation of the artificial noise.
                PSD,X,T=simulateddetectornoise(profile[det]
                                              ,windowSize
                                              ,fs,10,fs/2
                                              ,PSDm=PSDm[det]
                                              ,PSDc=PSDc[det])
                # Calculatint the psd of FFT=1s
                p, f = psd(X, Fs=fs,NFFT=fs)
                # Interpolate so that has t*fs values
                psd_int=interp1d(f,p)                                     
                PSD=psd_int(np.arange(0,fs/2,1/windowSize))
                PSD_dict[det]=PSD
                back_dict[det] = X
                # Making the noise a TimeSeries
                back=TimeSeries(X,sample_rate=fs) 
                # Calculating the ASD so tha we can use it for whitening later
                asd=back.asd(1,0.5)
                asd_dict[det] = asd
                gps_list.append(0.0)


            elif backgroundType == 'sudo_real':

                noise_seg=noise_segDict[det]
                # Calling the real noise segments
                noise=noise_seg[int(ind[det][I]):int(ind[det][I])+windowSize*fs]  
                # Generating the PSD of it
                p, f = psd(noise, Fs=fs, NFFT=fs) 
                p, f=p[1::],f[1::]
                # Feeding the PSD to generate the sudo-real noise.            
                PSD,X,T=simulateddetectornoise([f,p]
                                               ,windowSize,fs,10
                                               ,fs/2,PSDm=PSDm[det]
                                               ,PSDc=PSDc[det])
                p, f = psd(X, Fs=fs,NFFT=fs)
                # Interpolate so that has t*fs values
                psd_int=interp1d(f,p)                                     
                PSD=psd_int(np.arange(0,fs/2,1/windowSize))
                PSD_dict[det]=PSD
                back_dict[det] = X
                # Making the noise a TimeSeries
                back=TimeSeries(X,sample_rate=fs)
                # Calculating the ASD so tha we can use it for whitening later
                asd=back.asd(1,0.5)                 
                asd_dict[det] = asd
                gps_list.append(gps0[det]+ind[det][I]/fs+(windowSize-duration)/2)
                back_dict[det] = back.value

            elif backgroundType == 'real':
                noise_seg=noise_segDict[det]
                # Calling the real noise segments

                if noiseFormat == 'DataSet':
                    noise=noise_seg[I]

                else:
                    noise=noise_seg[int(ind[det][I]):int(ind[det][I])+windowSize*fs]
                # Calculatint the psd of FFT=1s
                p, f = psd(noise, Fs=fs,NFFT=fs)
                # Interpolate so that has t*fs values
                psd_int=interp1d(f,p)                                     
                PSD=psd_int(np.arange(0,fs/2,1/windowSize))
                PSD_dict[det]=PSD
                # Making the noise a TimeSeries
                back=TimeSeries(noise,sample_rate=fs)
                back_dict[det] = back
                # Calculating the ASD so tha we can use it for whitening later
                #print(det,back,len(back),type(back))
                asd=back.asd(1,0.5)
                asd_dict[det] = asd
                if noiseFormat == 'DataSet':
                    gps_list.append(gps0[det]+ind[det][I]+(windowSize-duration)/2)
                else:
                    gps_list.append(gps0[det]+ind[det][I]/fs+(windowSize-duration)/2)

            #If this dataset includes injections:            
            if injectionFolder != None:      

                # Calling the templates generated with PyCBC
                # OLD inj=load_inj(injectionFolder,injectionFileDict[det][inj_ind], det) 
                if inj_type =='oldtxt':
                    inj = np.loadtxt(injectionFolder+'/'
                                     +det+'/'+injectionFileDict[det][index_selection[det]])

                elif inj_type == 'directory':
                    inj_pod=DataPod.load(injectionFolder+'/'+injectionFileDict[det][index_selection[det]])
                    inj=np.array(inj_pod.strain[inj_pod.detectors.index(det)])


                elif inj_type == 'DataPod':
                    inj_pod=injectionFolder
                    inj=np.array(inj_pod.strain[inj_pod.detectors.index(det)])

                elif inj_type == 'DataSet':
                    inj_pod = injectionFileDict[det][index_selection[det]]
                    inj=np.array(inj_pod.strain[inj_pod.detectors.index(det)])

                if inj_type in ['DataSet','DataPod','directory']:

                    if injectionHRSS!=None:
                        if 'hrss' in inj_pod.pluginDict.keys():
                            hrss0=inj_pod.hrss
                        else:
                            raise AttributeError("There is no hrss in the injection pod")
                    else:
                        if 'hrss' in inj_pod.pluginDict.keys():
                            hrss0=inj_pod.hrss


                # Saving the length of the injection
                inj_len = len(inj)/fs
                inj_len_original=min(inj_len,duration)
                # I put a random offset for all injection so that
                # the signal is not always in the same place
                if inj_len > duration: 
                    print('Injection was cropped symmetricaly by '
                          +str(inj_len_original-fs*duration)+' to fit duration.')

                    inj = inj[int((inj_len-duration)*fs/2):]
                    inj_len_=len(inj)/fs                    
                    inj = inj[:int(duration*fs)]
                    inj_len = len(inj)/fs

                if inj_len <= duration:
                    inj = np.hstack((np.zeros(int(fs*(duration-inj_len)/2)), inj))
                    inj_len_=len(inj)/fs                    
                    inj = np.hstack((inj, np.zeros(int(fs*(duration-inj_len_)))))

                # DISPOSITIONS

                # Default case when coherent
                if disposition == 0:
                    if det == detKeys[0]:
                        disp = np.random.random_integers(low = min(-int(fs*(duration-inj_len_original)/2),0) 
                                                       ,high = max(int(fs*((duration-inj_len_original)/2 
                                                                     + injectionCrop*(duration))),1))

#                     # Case when signals will be randomly positioned for each detector seperatly. 
#                     # Does not affect if signals will be different or not.
#                     elif isinstance(disposition,(list,tuple):
#                         disp = np.random.random_integers(low = min(-int(fs*(duration-inj_len)/2),0) 
#                                                          ,high = max(int(fs*((duration-inj_len)/2 
#                                                                          + injectionCrop*(duration-0.1))),1))
                # Case when signals will be positioned within a duration stated by the disposition.
                # The center of this duration will be moved randomly given the maxDuration
                # of the injections.
                elif isinstance(disposition_,(int,float)) and disposition_>0:
                    if det == detKeys[0]:
                        wind = 0.2*disposition_/max((len(detKeys)-1),1)
                        center_position = np.random.random_integers(
                            low = min(-int(fs*(duration-maxDuration_-disposition_)/2),0)
                           ,high = max(int(fs*(duration-maxDuration_-disposition_)/2)+1,1))
                            # the +1 is because there is an error if the high becomes 0

                        if len(detKeys)>1:
                            positions=np.arange(-disposition_/2,disposition_/2+0.0001 
                                                ,disposition_/max((len(detKeys)-1),1))
                            positions=list([positions[0]+np.random.rand()*wind/2]
                                 +list(p+np.random.rand()*wind-wind/2 for p in positions[1:-1])
                                 +[positions[-1]-np.random.rand()*wind/2])

                        else:
                            positions=[0.0+np.random.rand()*wind/2]

                        np.random.shuffle(positions)
                    disp = center_position+int(positions[detKeys.index(det)]*fs)


                if disp >= 0: 
                    inj = np.hstack((np.zeros(int(fs*(windowSize-duration)/2)),inj[disp:]
                                         ,np.zeros(int(fs*(windowSize-duration)/2)+disp)))   
                if disp < 0: 
                    inj = np.hstack((np.zeros(int(fs*(windowSize-duration)/2)-disp),inj[:disp]
                                         ,np.zeros(int(fs*(windowSize-duration)/2)))) 


                # Calculating the one sided fft of the template,
                # Norm default is 'backwards' which means that it normalises with 1/N during IFFT and not duriong FFT
                inj_fft_0=(1/fs)*np.fft.fft(inj)
                inj_fft_0_dict[det] = inj_fft_0

                # Getting rid of negative frequencies and DC
                inj_fft_0N= inj_fft_0[1:int(windowSize*fs/2)+1]


                SNR0_dict[det]=np.sqrt((1/windowSize #(fs/N=fs/fs*windowSize 
                                       )*4*np.sum( # 2 from PSD, S=1/2 S1(one sided) and 2 from integral symetry
                    np.abs(inj_fft_0N*inj_fft_0N.conjugate())[windowSize*fl-1:windowSize*fm-1]
                                               /PSD_dict[det][windowSize*fl-1:windowSize*fm-1]))

                if single == True:
                    if det != luckyDet:
                        SNR0_dict[det] = 0.01 # Making single detector injections as glitch
                        inj_fft_0_dict[det] = np.zeros(windowSize*fs)


            else:
                SNR0_dict[det] = 0.01 # avoiding future division with zero
                inj_fft_0_dict[det] = np.zeros(windowSize*fs)


#             print(
#                 list(
#                     len(np.where(
#                         noise_segDict[det][int(ind[det][I]):int(ind[det][I])+windowSize*fs]
#                     )[0]
#                        )
#                     for det in detKeys
#                 )
#             )
        if (backgroundType=='real' 
            and any(len(np.where(noise_segDict[det][
                int(ind[det][I]):int(ind[det][I])+windowSize*fs]==0)[0])==windowSize*fs for det in detKeys)):
            print(I,"skipped")
            continue

        # Calculation of combined SNR    
        SNR0=np.sqrt(np.sum(np.asarray(list(SNR0_dict.values()))**2))

        # Tuning injection amplitude to the SNR wanted
        podstrain = []
        podPSD = []
        podCorrelations=[]
        SNR_new=[]
        SNR_new_sq_sum=0
        psdPlugDict={}
        plugInToApply=[]

        for det in detectors:
            if injectionHRSS!=None:
                fft_cal=(injectionHRSS/hrss0)*inj_fft_0_dict[det]         
            else:
                fft_cal=(injectionSNR/SNR0)*inj_fft_0_dict[det] 
            # Norm default is 'backwards' which means that it normalises with 1/N during IFFT and not duriong FFT
            if ignoreDetector ==None:
                inj_cal=np.real(np.fft.ifft(fs*fft_cal)) 
            elif ignoreDetector==det:
                inj_cal=0.0001*np.real(np.fft.ifft(fs*fft_cal)) 
            else:
                inj_cal=np.real(np.fft.ifft(fs*fft_cal)) 
            # Joining calibrated injection and background noise
            strain=TimeSeries(back_dict[det]+inj_cal,sample_rate=fs,t0=0).astype('float64')
            #print(det,len(strain),np.prod(np.isfinite(strain)),len(strain)-np.sum(np.isfinite(strain)))
            #print(det,len(strain),'zeros',len(np.where(strain.value==0.0)[0]))
            #print(strain.value.tolist())
            # Bandpassing
            # strain=strain.bandpass(20,int(fs/2)-1)
            # Whitenning the data with the asd of the noise
            whiten_strain=strain.whiten(4,2,fduration=4,method = 'welch', highpass=20)#,asd=asd_dict[det])

            #print(det,len(strain),np.prod(np.isfinite(strain)),len(strain)-np.sum(np.isfinite(strain)))
            #print(det,len(strain),'zeros',len(np.where(strain.value==0.0)[0]))

            # Crop data to the duration length
            whiten_strain=whiten_strain[int(((windowSize-duration)/2)*fs):int(((windowSize+duration)/2)*fs)]
            podstrain.append(whiten_strain.value.tolist())

            if 'snr' in plugins:
                whiten_strain_median=strain.whiten(4,2,fduration=4,method = 'median'
                        , highpass=20)[int(((windowSize-duration)/2)*fs):int(((windowSize+duration)/2)*fs)]
                # This is strictly for duration 1 and fs 1024
                new_snr = np.sum(whiten_strain_median.value**2 -0.978**2)

                SNR_new.append(np.sqrt(max(new_snr,0)))
                SNR_new_sq_sum += new_snr


            # if 'snr' in plugins:
            #     # Adding snr value as plugin.
            #     # Calculating the new SNR which will be slightly different that the desired one.    
            #     inj_fft_N=fft_cal[1:int(windowSize*fs/2)+1]
            #     SNR_new.append(np.sqrt((1/windowSize #(fs/fs*windowsize
            #                            )*4*np.sum( # 2 from integral + 2 from S=1/2 S1(one sided)
            #         np.abs(inj_fft_N*inj_fft_N.conjugate())[windowSize*fl-1:windowSize*fm-1]
            #                                  /PSD_dict[det][windowSize*fl-1:windowSize*fm-1])))

            #     plugInToApply.append(PlugIn('snr'+det,SNR_new[-1]))

            if 'psd' in plugins:
                podPSD.append(asd_dict[det]**2)

        if 'snr' in plugins: 

            network_snr = np.sqrt(max(SNR_new_sq_sum,0))
            SNR_new.append(network_snr)
            plugInToApply.append(PlugIn('snr',SNR_new))




        if 'psd' in plugins:
            plugInToApply.append(PlugIn('psd'
                                        ,genFunction=podPSD
                                        ,plotFunction=plotpsd
                                        ,plotAttributes=['detectors','fs']))

        for pl in plugins:
            if 'correlation' in pl:
                plugInToApply.append(knownPlugIns(pl))



        pod = DataPod(strain = podstrain
                           ,fs = fs
                           ,labels =  labels
                           ,detectors = detKeys
                           ,gps = gps_list
                           ,duration = duration)

        if injectionFolder!=None and inj_type in ['DataSet','DataPod','directory']:

            for plkey in list(inj_pod.pluginDict.keys()):
                if not (plkey in list(pod.pluginDict.keys())):
                    pod.addPlugIn(inj_pod.pluginDict[plkey])

        if 'hrss' in plugins: 
            if 'hrss' in inj_pod.pluginDict.keys():
                if injectionHRSS!=None:
                    plugInToApply.append(PlugIn('hrss'
                                            ,genFunction=inj_pod.hrss*(injectionHRSS/hrss0)))
                else:
                    plugInToApply.append(PlugIn('hrss'
                                            ,genFunction=inj_pod.hrss*(injectionSNR/SNR0)))

            else:
                print("Warning: Unable to calculate hrss, There was no hrss in the injection pod.")

        for pl in plugInToApply:
            pod.addPlugIn(pl)

        DATA.add(pod)

        #t1=time.time()
        #sys.stdout.write("\r Instantiation %i / %i --- %s" % (I+1, size, str(t1-t0)))
        #sys.stdout.flush()
        #t0=time.time()

    if not ('shuffle' in kwargs):
        kwargs['shuffle'] = True

    if not isinstance(kwargs['shuffle'],bool):
        raise TypeError("shuffle option must be a bool value")

    if kwargs['shuffle']==True:
         random.shuffle(DATA.dataPods)
    # else if false do not shuffle.

    print('\n')
    if savePath!=None:
        DATA.save(savePath+'/'+name,'pkl')
    else:
        return(DATA)
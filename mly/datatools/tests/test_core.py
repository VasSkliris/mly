import numpy as np
from mly.core import *

class ΤestDataPodBase:
    
    TEST_CLASS=DataPodBase

    # --> function to help creating the 
    #     when needed
    
    @classmethod
    def create(*args,**kwargs):
        return cls.TEST_CLASS(*args,**kwargs)
    
    # --> test calling an empty class
   
    @property
    def TEST_CLASS(self):
        return self._TEST_ARRAY
    
    # --> test different cases of calling
    #     the class and creating it
    
    def test_new(self):
        """Test DataPod Creation
        """
        
        # --> test empty datapod
        
        with pytest.raises(TypeError):
            self.TEST_CLASS()
            
        # test unequal strain data
        
        with pytest.raises(IndexError):
            self.create([np.random.randn(256)
                         ,np.random.randn(251)])
        with pytest.raises(IndexError):
            self.create([np.random.randn(256)
                         ,np.random.randn(251)
                         ,np.random.randn(221)])
        
        # --> test only with strain data
        
        for d in range(1,5):
            size=np.random.randint(111,222)
            pod=self.create(np.random.randn(d,size))
            assert pod.shape == (d,size)
            assert len(pod.detectors)==d
            assert pod.fs==1
            assert pod.duration ==size/pod.fs
            
        # --> test that error is raised with inf
        #     or None 
        
        with pytest.raises(ValueError):
            strains=np.random.randn(2,230)
            strains[0][0]=np.inf
            pod=self.create(strains)
            
        # --> test the detectors to be appropriate values
        
        with pytest.raises(IndexError):
            self.create(np.random.randn(2,256),'H')
        with pytest.raises(IndexError):
            self.create(np.random.randn(2,256),'HLV')
        with pytest.raises(IndexError):
            self.create(np.random.randn(2,256),'NF')
            
        # --> test strain and detectors
        
        for d in ['H','U','HL','HLV','HLVK','HLVKI']:
            size=np.random.randint(111,222)
            pod=self.create(np.random.randn(len(d),size),d)
            assert pod.shape == (d,size)
            assert len(pod.detectors)==d
            assert pod.fs==1
            assert pod.duration ==size/pod.fs
            
        # --> test labels have right format
        
        with pytest.raises(TypeError):
            pod=self.create(np.random.randn(2,212)
                        ,detectors='HL'
                        ,labels=['noise'])
            
        pod=self.create(np.random.randn(2,212)
                        ,detectors='HL')
        assert isinstance(pod.labels,dict)
        assert len(pod.labels)==1
        
        pod=self.create(np.random.randn(2,212)
                        ,detectors='HL'
                        ,labels={'type':'noise'})
        assert isinstance(pod.labels,dict)
        assert len(pod.labels)==1
        
        pod=self.create(np.random.randn(2,212)
                        ,detectors='HL'
                        ,labels={'type':'noise'
                                 ,'horoscope':'possum'})
        assert isinstance(pod.labels,dict)
        assert len(pod.labels)==2
        assert 0
        
       

        
            
            

            
        
        
        
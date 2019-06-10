from mly.generators import *

from .__init__ import *
null_path=nullpath()

################################################################################
###### GENERATORS ##############################################################



# GENERATE A DATASET OF CBCS WITH REAL NOISE WITH SNR = 25

data_generator_cbc(parameters=['cbc_00','real',25]       
                       ,length=4           
                       ,fs = 2048              
                       ,size =10000            
                       ,detectors='HLV'  
                       ,spec=False
                       ,phase=False
                       ,res=128
                       ,noise_file=['20170825','SEG0_1187654416_2306s.txt']  
                       ,t=32             
                       ,lags=11
                       ,starting_point=200
                       ,name=''          
                       ,destination_path = null_path+'/datasets/cbc/'
                       ,demo=False)   





# GENERATE A DATASETS AS THE ABOVE FOR MANY SNR VALUES. THIS FUNCTION MANAGES
# ALL THE REAL NOISE DATA YOU HAVE SO THAT YOU USE THE MINIMUM YOU NEED.

auto_gen(set_type=['cbc','real','HLV','cbc_02',[60,40,30,25,20,16,12,10,8]]
             ,date_ini='20170808'
             ,size=10000
             ,fs=2048
             ,length=4 
             ,lags=1
             ,t=32
             ,s_name='')


# THIS FUNCTION HAS TO RUN AFTER THE auto_gen TO FINALISE YOUR DATASET FILE

finalise_gen(null_path+'/datasets/cbc/cbc_real1')


################################################################################
###### ML MODELS  ##############################################################




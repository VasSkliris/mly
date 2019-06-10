# -*- coding: utf-8 -*-

import os


def nullpath():
    pwd=os.getcwd()
    if 'MLy_Workbench' in pwd:
        null_path=pwd.split('MLy_Workbench')[0]+'MLy_Workbench'
    elif 'EMILY' in pwd:
        null_path=pwd.split('EMILY')[0]+'EMILY'
    else:
        null_path=''
        print('Warning: null_path is empty, you should run import mly, CreateMLyWorkbench()'
              +' to create a workbench or specify null_path value here to avoid FileNotFound errors.')
    return(null_path)

def CreateMLyWorkbench():
    
    if os.path.exists('MLy_Workbench'):
        print('MLy_Workbench already exists')
    else:
        os.system('mkdir MLy_Workbench')
            
    os.system('mkdir MLy_Workbench/datasets')
    os.system('mkdir MLy_Workbench/datasets/cbc')
    os.system('mkdir MLy_Workbench/datasets/noise') 
    os.system('mkdir MLy_Workbench/datasets/burst')
    os.system('mkdir MLy_Workbench/datasets/noise/optimal') 
    os.system('mkdir MLy_Workbench/datasets/noise/sudo_real') 
    os.system('mkdir MLy_Workbench/datasets/noise/real')
    os.system('mkdir MLy_Workbench/trainings') 
    os.system('mkdir MLy_Workbench/ligo_data') 
    os.system('mkdir MLy_Workbench/injections')
    os.system('mkdir MLy_Workbench/injections/cbcs') 
    os.system('mkdir MLy_Workbench/injections/bursts')

    print('Workbench is complete!')
    os.chdir('MLy_Workbench')
    
    null_path=nullpath()

    return

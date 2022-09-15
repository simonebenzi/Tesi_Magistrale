
import sys
import os
import glob

def LoadCarlaEggFileGivenPathToCarlaEggFolder(carlaEggPath):
    
    try:
        sys.path.append(glob.glob(carlaEggPath % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    except IndexError:
        pass
    
    return

def AddCarlaCodesToSysGivenCodeFolder(carlaCodesPath):
    
    try:
        sys.path.insert(0, carlaCodesPath) 
    except IndexError:
        pass
    
    return

# This works for Carla 0.9.10
def PrepareCarlaSysGivenCarlaFolder(carlaPath):
    
    carlaEggPath = carlaPath + 'PythonAPI/carla/dist/carla-*%d.%d-%s.egg'
    carlaCodesPath = carlaPath + 'PythonAPI/carla/'
    
    LoadCarlaEggFileGivenPathToCarlaEggFolder(carlaEggPath)
    AddCarlaCodesToSysGivenCodeFolder(carlaCodesPath)
    
    return

# This works for Carla 0.9.10
def PrepareCarlaSysGivenPathToFileWhereCarlaFolderIsNoted(filePathWhereCarlaFolderIsNoted):
    
    with open(filePathWhereCarlaFolderIsNoted) as f:
        carlaPath = f.readlines()[0]
        carlaPath = carlaPath.rstrip()
    
    PrepareCarlaSysGivenCarlaFolder(carlaPath)
    
    return
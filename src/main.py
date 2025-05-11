from primarySimFuncs import *
from dataAnalysis import *

def main():
    xVel, yVel, zVel, kx, ky, kz, times=runSim(12.0,129,1000.0,2*(10**-4.0))
    #plotKineticEnergyTime(xVel, yVel, zVel, times)
    plotEnergySpectrum1D(xVel, yVel, zVel, kx, ky, kz)
    return 0

if __name__=='__main__':
    main()
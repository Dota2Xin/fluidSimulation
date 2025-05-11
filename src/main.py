from primarySimFuncs import *
from dataAnalysis import *

def main():
    xVel, yVel, zVel, kx, ky, kz, times=runSim(2*np.pi,129,30.0,2*(10**-4.0))
    plotKineticEnergyTime(xVel, yVel, zVel, times)
    return 0

if __name__=='__main__':
    main()
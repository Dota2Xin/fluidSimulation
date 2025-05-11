from numba import njit
import numpy as np
import math
from initializationFuncs import *

def runSim(L, resolution, finalTime, nu):
    currentTime=0

    xVel = []
    yVel = []
    zVel = []

    kMax=2*np.pi*(resolution-1)/(2.0*L)
    dt=1/(nu*((kMax)**2.0))
    totalSteps=math.floor(finalTime/dt)
    times=np.linspace(0, finalTime, totalSteps)
    print(dt)
    print(totalSteps)
    kx, ky, kz, KX, KY, KZ, kSq, oldXVel, oldYVel, oldZVel, currXVel, currYVel, currZVel=createInitialData(L, resolution, nu, dt)

    xVel.append(oldXVel)
    yVel.append(oldYVel)
    zVel.append(oldZVel)

    xVel.append(currXVel)
    yVel.append(currYVel)
    zVel.append(currZVel)
    print(kSq[0][0][0])
    for i in range(totalSteps):
        print(i)
        newXVel, newYVel, newZVel=evolveVelocities(oldXVel, oldYVel, oldZVel, currXVel, currYVel, currZVel, KX, KY, KZ, kSq, dt, nu)
        oldXVel=np.copy(currXVel)
        oldYVel=np.copy(currYVel)
        oldZVel=np.copy(currZVel)

        currXVel=np.copy(newXVel)
        currYVel=np.copy(newYVel)
        currZVel=np.copy(newZVel)

        xVel.append(currXVel)
        yVel.append(currYVel)
        zVel.append(currZVel)

    return xVel, yVel, zVel, kx, ky ,kz, times

#previous velocity=v^(n-1)_k, current velocty=v^n_k
#kx,ky,kz just k arrays in each direction
#kSq, 3d array filled with value of k squared
#eta=viscocity, dt=timestep
def evolveVelocities(prevXVel, prevYVel, prevZVel, currXVel, currYVel, currZVel,KX,KY,KZ, kSq, dt, nu):
    firstPrefactor=np.exp(-2*nu*dt*kSq)
    secondPrefactor=-2*dt*np.exp(-nu*kSq*dt)

    firstTermX=prevXVel*firstPrefactor
    firstTermY=prevYVel*firstPrefactor
    firstTermZ=prevZVel*firstPrefactor

    Ax, Ay, Az=calcA(currXVel, currYVel, currZVel, KX,KY,KZ)

    cPrefactor=((Ax*KX+Ay*KY+Az*KZ)/kSq)
    Cx=Ax-cPrefactor*KX
    Cy=Ay-cPrefactor*KY
    Cz=Az-cPrefactor*KZ

    secondTermX=secondPrefactor*Cx
    secondTermY=secondPrefactor*Cy
    secondTermZ=secondPrefactor*Cz

    nextVelX=firstTermX+secondTermX
    nextVelY=firstTermY+secondTermY
    nextVelZ=firstTermZ+secondTermZ

    return nextVelX, nextVelY, nextVelZ

def calcA(xVel, yVel, zVel, KX,KY,KZ):
    Ax1=calcPseudoSpectral(xVel, xVel, KX)
    Ax2=calcPseudoSpectral(yVel, xVel, KY)
    Ax3=calcPseudoSpectral(zVel, xVel, KZ)
    Ax=Ax1+Ax2+Ax3

    Ay1 = calcPseudoSpectral(xVel, yVel, KX)
    Ay2 = calcPseudoSpectral(yVel, yVel, KY)
    Ay3 = calcPseudoSpectral(zVel, yVel, KZ)
    Ay = Ay1 + Ay2 + Ay3

    Az1 = calcPseudoSpectral(xVel, zVel, KX)
    Az2 = calcPseudoSpectral(yVel, zVel, KY)
    Az3 = calcPseudoSpectral(zVel, zVel, KZ)
    Az = Az1 + Az2 + Az3

    return Ax, Ay, Az

def calcPseudoSpectral(vel1, vel2, k):
    ifft1=np.fft.ifftn(vel1, norm='ortho')
    i=1j
    kScale=1/k[1][1][1]
    ifft2=np.fft.ifftn(i*vel2*k*kScale, norm='ortho')
    product=ifft1*ifft2
    result=np.fft.fftn(product, norm='ortho')
    return result
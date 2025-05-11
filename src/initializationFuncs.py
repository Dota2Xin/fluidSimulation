from numba import njit
import numpy as np
import primarySimFuncs

#resolution must be odd
def createInitialData(length, resolution, nu, dt):

    #create wavevectors in way numpy likes
    kx=np.fft.fftfreq(resolution, d=(length/resolution))*2*np.pi
    ky=np.fft.fftfreq(resolution, d=(length/resolution))*2*np.pi
    kz=np.fft.fftfreq(resolution, d=(length/resolution))*2*np.pi
    #numpy array manipulation
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    kSq=np.sqrt(KX**2+KY**2+KZ**2)
    kSq[0][0][0] = 1e6

    k0=(resolution/10.0)*(2*np.pi/(length))
    #initialize velocities
    initialXVel, initialYVel, initialZVel=initializeVelocities(kx,ky,kz,resolution,k0)

    #perform forward euler evalutation for dt to get two sets of initialVelocities separated by dt
    Ax, Ay, Az = primarySimFuncs.calcA(initialXVel, initialYVel, initialZVel, KX, KY, KZ)

    cPrefactor = ((Ax * KX + Ay * KY + Az * KZ) / kSq)
    Cx = Ax - cPrefactor * KX
    Cy = Ay - cPrefactor * KY
    Cz = Az - cPrefactor * KZ

    secondXVel=initialXVel-(nu*kSq*initialXVel)*dt-Cx*dt
    secondYVel = initialYVel - (nu * kSq * initialYVel)*dt - Cy*dt
    secondZVel = initialZVel - (nu * kSq * initialZVel)*dt - Cz*dt

    return kx, ky, kz, KX, KY, KZ, kSq, initialXVel, initialYVel, initialZVel, secondXVel, secondYVel, secondZVel



def initializeVelocities(kx,ky,kz, resolution, k0):
    initialXVel = np.zeros((resolution, resolution, resolution), dtype=complex)
    initialYVel = np.zeros((resolution, resolution, resolution), dtype=complex)
    initialZVel = np.zeros((resolution, resolution, resolution), dtype=complex)


    fullHalf = int((resolution - 1) / 2)
    for i in range(fullHalf+1):
        for j in range(fullHalf+1):
            for k in range(fullHalf+1):
                if i==0 and j==0 and k==0:
                    initialXVel[i][j][k]=0
                else:
                    kVec=np.asarray([kx[i],ky[j],kz[k]])

                    #force velocity perpendicular to k (divergence free condition)
                    if i==j==k:
                        e1Vec=np.asarray([kVec[1],-kVec[0],0])
                    else:
                        e1Vec=np.asarray([kVec[1]-kVec[2], kVec[2]-kVec[0], kVec[0]-kVec[1]])
                    e1Vec=e1Vec/np.linalg.norm(e1Vec)
                    e2Vec=np.cross(kVec, e1Vec)
                    e2Vec=e2Vec/np.linalg.norm(e2Vec)

                    #draw initial velocities from maxwell-boltzmann like distribution
                    amplitudeVec=np.random.normal(0, k0,3)
                    amplitude=np.sqrt(np.linalg.norm(amplitudeVec))
                    randomPhase1=2*np.pi*np.random.rand()
                    randomPhase2=2*np.pi*np.random.rand()

                    preFactor1=amplitude*np.exp(1j*randomPhase1)
                    preFactor2=amplitude*np.exp(1j*randomPhase2)

                    initialXVel[i][j][k]=preFactor1*e1Vec[0]+preFactor2*e2Vec[0]
                    initialYVel[i][j][k] = preFactor1 * e1Vec[1] + preFactor2 * e2Vec[1]
                    initialZVel[i][j][k] = preFactor1 * e1Vec[2] + preFactor2 * e2Vec[2]

                    #set v(-k)=v^*(k) to ensure velocity is real
                    iConj=i+(fullHalf-i+1)
                    jConj=j+(fullHalf-j+1)
                    kConj=k+(fullHalf-k+1)

                    initialXVel[iConj][jConj][kConj]=np.conjugate(initialXVel[i][j][k])
                    initialYVel[iConj][jConj][kConj] = np.conjugate(initialYVel[i][j][k])
                    initialZVel[iConj][jConj][kConj] = np.conjugate(initialZVel[i][j][k])
    return initialXVel, initialYVel, initialZVel


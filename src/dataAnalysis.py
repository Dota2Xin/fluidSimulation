from numba import njit
import numpy as np
import matplotlib.pyplot as plt
import math

def plotKineticEnergyTime(xVel, yVel, zVel, times):
    energies = []
    for i in range(len(times)):
        e = computeTotalKineticEnergy(xVel[i], yVel[i], zVel[i])
        energies.append(e)

    plt.plot(times, energies)
    plt.xlabel("Time")
    plt.ylabel("Total Kinetic Energy")
    plt.title("Kinetic Energy Decay (Sanity Check)")
    plt.grid(True)
    plt.show()

def computeTotalKineticEnergy(xVel, yVel, zVel):
    # Fourier space: energy is 0.5 * sum(|v_x|^2 + |v_y|^2 + |v_z|^2)
    energy = 0.5 * np.sum(np.abs(xVel)**2 + np.abs(yVel)**2 + np.abs(zVel)**2)
    return energy
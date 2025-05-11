from numba import njit
import numpy as np
import matplotlib.pyplot as plt
import math

def plotKineticEnergyTime(xVel, yVel, zVel, times):
    energies = []
    for i in range(len(times)):
        e = computeTotalKineticEnergy(xVel[i], yVel[i], zVel[i])
        energies.append(e)
    plt.yscale('log')
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


def plotEnergySpectrum1D(xVel, yVel, zVel, kx, ky, kz):
    finalX = xVel[-1]
    finalY = yVel[-1]
    finalZ = zVel[-1]

    resolution = finalX.shape[0]

    # Compute total energy at each point in k-space
    energyDensity = 0.5 * (np.abs(finalX)**2 + np.abs(finalY)**2 + np.abs(finalZ)**2)

    # Make full 3D k-grid
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    kMag = np.sqrt(KX**2 + KY**2 + KZ**2)

    # Bin energy into spherical shells
    kMax = np.max(kMag)
    numBins = resolution // 2
    binEdges = np.linspace(0, kMax, numBins + 1)
    kBinCenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    energySpectrum = np.zeros(numBins)
    binCounts = np.zeros(numBins)

    flatK = kMag.ravel()
    flatE = energyDensity.ravel()
    binIndices = np.digitize(flatK, binEdges) - 1

    for i in range(flatK.size):
        binIndex = binIndices[i]
        if 0 <= binIndex < numBins:
            energySpectrum[binIndex] += flatE[i]
            binCounts[binIndex] += 1

    # Normalize by number of modes per bin
    nonzero = binCounts > 0
    energySpectrum[nonzero] /= binCounts[nonzero]

    log_k = np.log((kBinCenters[nonzero])[5:])
    log_E = np.log((energySpectrum[nonzero])[5:])

    # Linear fit in log-log space
    slope, intercept = np.polyfit(log_k, log_E, 1)
    print(f"Slope: {slope:.4f}")

    # Generate fitted line
    fit_line = np.exp(intercept) * ((kBinCenters[nonzero])[5:]) ** slope

    # Plotting
    plt.figure()

    plt.loglog((kBinCenters[nonzero])[5:], (energySpectrum[nonzero])[5:])
    plt.loglog((kBinCenters[nonzero])[5:], fit_line, '--', label=f"Slope = {slope:.2f}")
    plt.xlabel("Wavenumber |k|")
    plt.ylabel("Energy Spectrum E(k)")
    plt.title("1D Spherically Averaged Energy Spectrum at Final Time")
    plt.grid(True)
    plt.show()
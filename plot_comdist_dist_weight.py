import numpy as np
import matplotlib.pyplot as plt
import os

nrep = 288
rep_ini = 1
rep_end = 36
bins = 40
gap = 100


def main():
    dat = []
    wt = []
    for repi in range(rep_ini, rep_end + 1):
        datname = f'../16_mbar_reus/input/sort_comdist/para{repi}.comdis'
        dat_i = np.loadtxt(datname)[::gap, 1]
        dat.append(dat_i)

        wtname = f'../16_mbar_reus/result/fes_36/output{repi}.weight'
        wt_i = np.loadtxt(wtname)[::gap, 1]
        wt.append(wt_i)

    dat = np.hstack(dat)
    wt = np.hstack(wt)
    prob, x_edges = np.histogram(dat, density=True, bins=bins, weights=wt)
    x = (x_edges[1:] + x_edges[:-1]) / 2
    plt.plot(x, prob)
    plt.xlabel(r'Abl-abltide distance($\AA$)')
    plt.ylabel('Probability')
    plt.ylim(0, 0.06)
    outname = 'picture/comdist_weight.png'
    os.makedirs(os.path.dirname(outname), exist_ok=True)
    plt.savefig(outname, dpi=300)

    



if __name__ == '__main__':
    main()

import mdtraj as md
import numpy as np
from cluster import Cluster


def main():
    nrep = 288
    rep_ini = 1
    rep_end = 36
    bins = 40
    gap = 100
    dat = []
    wt = []
    for repi in range(rep_ini, rep_end + 1):
        datname = f'../16_mbar_reus/input/sort_comdist/para{repi}.comdis'
        dat_i = np.loadtxt(datname)[::gap, 1]
        # print(dat_i.shape)
        dat.append(dat_i)

        # wtname = f'../16_mbar_reus/result/fes_36/output{repi}.weight'
        # wt_i = np.loadtxt(wtname)[::gap, 1]
        # wt.append(wt_i)

    dat = np.hstack(dat)
    # wt = np.hstack(wt)

    kc = Cluster(dat, k=2, outfile='output_unweight_python.idx', stop_threshold=100, weights=None)


if __name__ == '__main__':
    main()

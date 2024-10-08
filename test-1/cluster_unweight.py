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
        datname = f'../../16_mbar_reus/input/sort_comdist/para{repi}.comdis'
        dat_i = np.loadtxt(datname)[::gap, 1]
        dat.append(dat_i)

    dat = np.hstack(dat)

    kc = Cluster(dat, k=2, outfile='output_unweight_python.idx', stop_threshold=100, weights=None)


if __name__ == '__main__':
    main()

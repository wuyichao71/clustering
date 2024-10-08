import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import os

nrep = 288
rep_ini = 1
rep_end = 36
bins = 40
gap = 100
topname = 'comdist_top.pdb'
colors = ['r', 'b', 'g']
markers = ['x', '^']

def plot_distribution(dat, wt):
    prob, x_edges = np.histogram(dat, density=True, bins=bins, weights=wt)
    x = (x_edges[1:] + x_edges[:-1]) / 2
    plt.plot(x, prob, color='k', lw=3)
    plt.xlabel(r'Abl-abltide distance($\AA$)')
    plt.ylabel('Probability')
    plt.ylim(0, 0.06)


def plot_kmeans_center(dat, weight, k_list, ax):
    cnt_list = []
    for k in k_list:
        cnt_list.append(np.average(dat[k], weights=weight[k]))
    print(cnt_list)
    arg = np.argsort(cnt_list)
    ylim = ax.get_ylim()
    plt.vlines(np.array(cnt_list)[arg], ylim[0], ylim[1], colors=colors[:len(k_list)])
    spt = np.mean(cnt_list)
    plt.vlines(spt, ylim[0], ylim[1], colors='k')
    return arg


def plot_point(dat, k_list, arg, ax):
    ylim = ax.get_ylim()
    for i, ti in enumerate(arg):
        y = np.random.uniform(low=ylim[0], high=ylim[1], size=len(dat[k_list[ti]]))
        plt.plot(dat[k_list[ti]], y, ls='', marker=markers[i], color=colors[i], ms=0.1)
    

def get_dat():
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
    return dat, wt
    

def get_idx(idxname, k):
    k_list = []
    idx = np.loadtxt(idxname)[:, 1]
    for ki in range(1, k + 1):
        k_list.append(idx == ki)
    return k_list


def get_traj_dat(dat, k_list):
    traj_dat = []
    for k_index in k_list:
        traj_dat.append(dat[k_index])
    return traj_dat


def main():
    dat, wt = get_dat()
    k_list = get_idx('output_python.idx', 2)
    plot_distribution(dat, wt)
    ax = plt.gca()
    arg = plot_kmeans_center(dat, wt, k_list, ax)
    plot_point(dat, k_list, arg, ax)
    outname = 'picture/comdist_wtkmeans_weight.png'
    os.makedirs(os.path.dirname(outname), exist_ok=True)
    plt.savefig(outname, dpi=300)

    



if __name__ == '__main__':
    main()

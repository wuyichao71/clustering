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
traj_list = ['output_1.dcd', 'output_2.dcd']
colors = ['r', 'b', 'g']
markers = ['x', '^']

def plot_distribution(dat):
    prob, x_edges = np.histogram(dat, density=True, bins=bins)
    x = (x_edges[1:] + x_edges[:-1]) / 2
    plt.plot(x, prob, color='k', lw=3)
    plt.xlabel(r'Abl-abltide distance($\AA$)')
    plt.ylabel('Probability')
    plt.ylim(0, 0.06)


def plot_kmeans_center(traj_dat, ax):
    cnt_list = []
    for dat in traj_dat:
        cnt_list.append(dat.mean())
    print(cnt_list)
    arg = np.argsort(cnt_list)
    ylim = ax.get_ylim()
    plt.vlines(np.array(cnt_list)[arg], ylim[0], ylim[1], colors=colors[:len(traj_dat)])
    spt = np.mean(cnt_list)
    plt.vlines(spt, ylim[0], ylim[1], colors='k')
    return arg


def plot_point(traj_dat, arg, ax):
    ylim = ax.get_ylim()
    for i, ti in enumerate(arg):
        y = np.random.uniform(low=ylim[0], high=ylim[1], size=len(traj_dat[ti]))
        plt.plot(traj_dat[ti], y, ls='', marker=markers[i], color=colors[i], ms=0.1)
    
    
def get_traj(traj_list):
    dat = []
    for trajname in traj_list:
        traj = md.load(trajname, top=topname)
        dat.append(traj.xyz[:, 0, 0] * 10)
    return dat
    

def get_dat():
    dat = []
    for repi in range(rep_ini, rep_end + 1):
        datname = f'../16_mbar_reus/input/sort_comdist/para{repi}.comdis'
        dat_i = np.loadtxt(datname)[::gap, 1]
        dat.append(dat_i)
    dat = np.hstack(dat)
    return dat


def get_idx(idxname, k):
    k_list = []
    idx = np.loadtxt(idxname)[:, 1]
    for ki in range(1, k + 1):
        k_list.append(idx == ki)
    return k_list


def main():
    dat = get_dat()
    # k_list = get_idx('output.idx', len(traj_list))
    traj_dat = get_traj(traj_list)
    plot_distribution(dat)
    ax = plt.gca()
    arg = plot_kmeans_center(traj_dat, ax)
    plot_point(traj_dat, arg, ax)
    outname = 'picture/comdist_unwtkmeans_unweight.png'
    os.makedirs(os.path.dirname(outname), exist_ok=True)
    plt.savefig(outname, dpi=300)

    



if __name__ == '__main__':
    main()

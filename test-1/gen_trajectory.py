import mdtraj as md
import numpy as np

nrep = 288
rep_ini = 1
rep_end = 36
gap = 100

def load_dat():
    dat = []
    for repi in range(rep_ini, rep_end + 1):
        datname = f'../16_mbar_reus/input/sort_comdist/para{repi}.comdis'
        dat_i = np.loadtxt(datname)[::gap, 1]
        dat.append(dat_i)
    return np.hstack(dat)

def main():
    top = md.Topology()
    chain = top.add_chain()
    residue = top.add_residue('ALA', chain)
    elem = md.core.element.get_by_symbol('C')
    top.add_atom('CA', elem, residue)

    dat = load_dat()
    xyz = np.zeros((len(dat), 1, 3))
    xyz[:, 0, 0] = dat / 10
    traj = md.Trajectory(xyz, top)
    print(traj)
    # traj[0].save('comdist_top.pdb')
    # traj[0].save('comdist_top.psf')
    # traj.save('comdist.dcd')


if __name__ == '__main__':
    main()

import mdtraj as md
import numpy as np


class Cluster():
    def __init__(self, features, k, outfile, weights=None, method='kmeans', initial_method='random_choice_center', max_iter=20):
        if features.ndim == 1:
            self.features = features.reshape((-1, 1))
        else:
            self.features = features
        self.weights = weights
        self.method = method
        self.k = k
        self.initial_method = initial_method
        self.max_iter = max_iter
        self.outfile = outfile

        self.index = None
        self.center = None

        if self.initial_method == 'random_choice_center':
            self.random_choice_center()

        if self.method == 'kmeans':
            self.kmeans()

        self.output()

    def random_choice_center(self):
        np.random.seed(0)
        center_index = np.random.choice(len(self.features), size=self.k, replace=False)
        center = self.features[center_index]
        self.center = np.array(center)

    def kmeans(self):
        for i in range(self.max_iter):
            # old_index = self.index
            # old_center = self.center

            # assign index
            dist = []
            for ki in range(self.k):
                dist_i = np.sum((self.features - self.center[ki])**2, axis=-1)
                dist.append(dist_i)
            dist = np.array(dist)
            self.index = np.argmin(dist, axis=0)

            # assign center
            center = []
            for ki in range(self.k):
                examples = self.index == ki
                features_i = self.features[examples]
                if self.weights is None:
                    center_i = np.average(features_i, axis=0)
                else:
                    weights_i = self.weights[examples]
                    center_i = np.average(features_i, weights=weights_i, axis=0)
                center.append(center_i)
            self.center = np.array(center)
        
    def output(self):
        with open(self.outfile, 'w') as out:
            for ii, index in enumerate(self.index):
                out.write(f'{ii+1:10d} {index+1:10d}\n')


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
        dat.append(dat_i)

        wtname = f'../16_mbar_reus/result/fes_36/output{repi}.weight'
        wt_i = np.loadtxt(wtname)[::gap, 1]
        wt.append(wt_i)

    dat = np.hstack(dat)
    wt = np.hstack(wt)

    kc = Cluster(dat, k=2, outfile='output_python.idx', weights=wt)


if __name__ == '__main__':
    main()

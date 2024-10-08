import mdtraj as md
import numpy as np


class Cluster():
    def __init__(self, features, k, outfile,
            weights=None, method='kmeans',
            initial_method='random_choice_center', 
            max_iter=20,
            stop_threshold=98):
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
        self.stop_threshold = stop_threshold

        self.index = None
        self.center = None

        # if self.initial_method == 'random_choice_center':
        #     self.random_choice_center()
        # if self.initial_method == 'random_choice_index':
        #     self.random_choice_index()
        # if self.initial_method == 'initial.input':
        #     self.initial_input()
        getattr(self, self.initial_method)()

        # DEBUG:
        print('------------------------')
        print(-1)
        print('self.index:', self.index)
        print('------------------------')

        if self.method == 'kmeans':
            self.kmeans()

        self.output()

    def random_choice_center(self):
        # DEBUG:
        np.random.seed(0)
        center_index = np.random.choice(len(self.features), size=self.k, replace=False)
        center = self.features[center_index]
        self.center = np.array(center)

    def random_generate_index(self):
        # DEBUG:
        np.random.seed(0)
        self.index = np.random.randint(self.k, size=len(self.feature))

    # for debug:
    def initial_input(self):
        pass


    def kmeans(self):
        diff1 = None
        diff2 = None
        for i in range(self.max_iter):
            old_index = self.index
            old_center = self.center

            # assign index
            if self.center is not None:
                dist = []
                for ki in range(self.k):
                    dist_i = np.sum((self.features - self.center[ki])**2, axis=-1)
                    dist.append(dist_i)
                dist = np.array(dist)
                self.index = np.argmin(dist, axis=0)

            # assign center
            if self.index is not None:
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

            ndata = np.zeros(self.k)
            for ki in range(self.k):
                ndata[ki] = np.sum(self.index == ki)

            if old_index is None:
                converged = False
            else:
                diff1 = diff2
                diff2 = np.full(self.k, np.nan)
                for ki in range(self.k):
                    pos = self.index == ki
                    diff2[ki] = np.sum(old_index[pos] != self.index[pos])
                if diff1 is not None:
                    converged = True
                    convergency = np.min(100 - 100 * np.abs(diff2 - diff1) / ndata)
                    if convergency < self.stop_threshold:
                        converged = False
                else:
                    converged = False

            # DEBUG:
            print('------------------------')
            print(i)
            print('old_index:', old_index)
            print('self.index:', self.index)
            print('diff1:', diff1)
            print('diff2:', diff2)
            print('------------------------')

            if converged:
                break


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
        # print(dat_i.shape)
        dat.append(dat_i)

        wtname = f'../16_mbar_reus/result/fes_36/output{repi}.weight'
        wt_i = np.loadtxt(wtname)[::gap, 1]
        wt.append(wt_i)

    dat = np.hstack(dat)
    wt = np.hstack(wt)

    kc = Cluster(dat,
            k=2,
            outfile='output_python.idx',
            initial_method='random_generate_index',
            input_index_file='input.idx'
            weights=wt)


if __name__ == '__main__':
    main()

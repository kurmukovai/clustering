import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from scipy.signal import argrelmin
from sklearn.metrics import adjusted_rand_score
import time

"""
class parameters : 

    random_state
    X
    y
    max_label
    iter
    xpca
    history
    
history has the following structure : 
 ~n_iteration x [xmin, ymin, self.iter, parent] (iter x 4 ndarray)
 where row index stands for label of cluster on the current hierarchy level
 with xmin, ymin, iteration of obtaining and parent cluster

class methods :
    
    __init__
    fit
    _density_est
    _which_split
    _is_stop
    _split
"""


class dePDDP():
    """
    Boley, Daniel. "Principal direction divisive partitioning."
    Data mining and knowledge discovery 2.4 (1998): 325-344.
    
    Splitting is done in a top-down manner
    under assumpsion that minima of the (density distribution) first principle
    component provide best split for the data (same idea used for choosing 
    cluster to split and as termination criterium)
    
    
    USE
    
    depddp = dePDDP()
    y_pred = depddp.fit(x)
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X):

        self.X = X - X.mean(axis=0)  # z-scoring?

        self.y = np.zeros((1, X.shape[0]))
        self.max_label = 1
        self.iter = 0
        self.xpca = []
        parent = -1

        xpca, xmin, ymin = self._density_est(self.X)
        self.history = np.array([[xmin, ymin, self.iter, parent]], dtype=float)
        self.xpca.append(xpca)

        while not self._is_stop():
            parent = self._which_split()  # parent is an index, not data
            y = self._split(parent)
            self.y = np.concatenate((self.y, y), axis=0)

            self.iter += 1
            self.max_label = self.history.shape[0]

        return self.y[-1]

    def _split(self, parent):
        """
        input:
            parent label in self.y of the cluster to split
        output:
            y 1 x total_n_objects ndarray  
        """
        # init params, global y, local _y, params of the parent cluster,
        # labels for child clusters

        y = self.y[-1].copy()
        parent_xpca = self.xpca[parent]
        parent_xmin = self.history[parent, 0]
        l_label = 0 + self.max_label
        r_label = 1 + self.max_label

        # split parent cluster
        _y = np.where(parent_xpca <= parent_xmin, l_label, r_label)
        y[np.where(y == parent)] = _y[:, 0]

        # precompute information for child clusters
        l_x = self.X[np.where(y == l_label)]
        r_x = self.X[np.where(y == r_label)]
        l_xpca, l_xmin, l_ymin = self._density_est(l_x)
        r_xpca, r_xmin, r_ymin = self._density_est(r_x)

        # store information for child clusters
        l_history = np.array([[l_xmin, l_ymin, self.iter, parent]], dtype=float)
        r_history = np.array([[r_xmin, r_ymin, self.iter, parent]], dtype=float)
        self.history = np.concatenate((self.history, l_history), axis=0)
        self.history = np.concatenate((self.history, r_history), axis=0)
        self.xpca.append(l_xpca)
        self.xpca.append(r_xpca)

        # zerod parent row since it is already divided
        self.history[parent, 0], self.history[parent, 1] = np.nan, np.nan

        return y.reshape(1, y.shape[0])

    def _density_est(self, x):
        """
        input: 
            x n_objects x n_features ndarray
            
        output:
            xpca cluster projected onto its first principal component
            xmin x value of minimal minima of produced kernel density
            ymin y value of minimal minima of produced kernel density
        """
        if x.shape[0] < 3: return np.nan, np.nan, np.nan

        pca = PCA(n_components=1, random_state=self.random_state)
        kde = KernelDensity()

        xpca = pca.fit_transform(x)
        h = np.std(xpca) * (4 / 3 / len(xpca)) ** (1 / 5)
        kde.set_params(bandwidth=h).fit(xpca)

        mmin, mmax = np.percentile(xpca, [5, 95])
        xdensity = np.linspace(mmin, mmax, 1000)[:, np.newaxis]
        ydensity = np.exp(kde.score_samples(xdensity))  # think about this 1000
        # number 1000 seems pretty arbitrary for me
        local_minimas_idx = argrelmin(ydensity)[0]
        if local_minimas_idx.size == 0:
            return xpca, np.nan, np.nan
        else:
            idx = ydensity[local_minimas_idx].argmin()
            xmin = xdensity[local_minimas_idx[idx]]
            ymin = ydensity[local_minimas_idx[idx]]

        return xpca, xmin, ymin

    def _is_stop(self, ):
        """
        output:
            True if terminal criterium is met, False o/w
        """
        return np.all(np.isnan(self.history[:, 0]))

    def _which_split(self, ):
        """
        output:
            label in self.y, which cluster to split on current iteration
        """
        return np.nanargmin(self.history[:, 1])


class ward_based_pdb():
    """
    Ward distance based approach.
    Compute best(bigger-better) ward distance for all existent clusters,
    (to be precise for their projections on the first principle component)
    choose for splitting one with the biggest one, stop when either 
     prespecified number of clusters is reached, or when there is no minimum
     of density distribution for all existent clusters (this is similar to depddp)
    
    """

    def __init__(self, n_clusters, random_state=42, stopping='n_clusters'):
        self.random_state = random_state
        self.n_clusters = n_clusters
        self.stopping = stopping

    def fit(self, X):

        self.X = X - X.mean(axis=0)  # z-scoring?

        self.y = np.zeros((1, X.shape[0]))
        self.max_label = 1
        self.iter = 0
        self.xpca = []
        parent = -1

        xpca, xmax, max_ward, _ = self._ward_est(self.X)

        self.history = np.array([[xmax, max_ward, self.iter, parent, True]], dtype=float)  # first step we always split!
        self.xpca.append(xpca)

        flag = not self._is_stop()
        while flag:

            parent = self._which_split()  # parent is an index, not data
            y = self._split(parent)
            self.y = np.concatenate((self.y, y), axis=0)

            self.iter += 1
            self.max_label = self.history.shape[0]
            flag = not self._is_stop()
            if self.y.shape[0] == self.n_clusters and self.stopping == 'n_clusters':
                flag = False

        return self.y[-1]

    def _split(self, parent):
        """
        input:
            parent label in self.y of the cluster to split
        output:
            y 1 x total_n_objects ndarray  
        """
        # init params, global y, local _y, params of the parent cluster,
        # labels for child clusters

        y = self.y[-1].copy()
        parent_xpca = self.xpca[parent]
        parent_xmax = self.history[parent, 0]
        l_label = 0 + self.max_label
        r_label = 1 + self.max_label

        #  split parent cluster
        _y = np.where(parent_xpca <= parent_xmax, l_label, r_label)
        y[np.where(y == parent)] = _y[:, 0]

        # precompute information for child clusters
        l_x = self.X[np.where(y == l_label)]
        r_x = self.X[np.where(y == r_label)]
        l_xpca, l_xmax, l_wd, l_flag = self._ward_est(l_x)
        r_xpca, r_xmax, r_wd, r_flag = self._ward_est(r_x)

        # store information for child clusters
        l_history = np.array([[l_xmax, l_wd, self.iter, parent, l_flag]], dtype=float)
        r_history = np.array([[r_xmax, r_wd, self.iter, parent, r_flag]], dtype=float)
        self.history = np.concatenate((self.history, l_history), axis=0)
        self.history = np.concatenate((self.history, r_history), axis=0)
        self.xpca.append(l_xpca)
        self.xpca.append(r_xpca)

        # zerod parent row since it is already divided
        self.history[parent, 0], self.history[parent, 1], self.history[parent, 4] = np.nan, np.nan, False

        return y.reshape(1, y.shape[0])

    def _ward_est(self, x):
        """
        input: 
            x n_objects x n_features ndarray
            
        output:
            xpca cluster projected onto its first principal component
            xmax x value optimal split (in terms of Ward distance)
            max_ward value of Ward distance for xmax split 
            flag True/False, if xpca has local minima True, False o/w   
        """

        if x.shape[0] < 3:
            return np.nan, np.nan, np.nan, False

        pca = PCA(n_components=1, random_state=self.random_state)

        xpca = pca.fit_transform(x)
        mmin, mmax = np.percentile(xpca, [20, 80])  # xpca.min(), xpca.max(), np.percentile(xpca, [20, 80])
        xgrid = np.linspace(mmin, mmax, 1000)[:, np.newaxis]
        ward_distance = []

        for xsplit in xgrid:
            _left, _right = xpca[xpca <= xsplit], xpca[xpca > xsplit]
            c_left, c_right = _left.mean(), _right.mean()
            n_left, n_right = _left.shape[0], _right.shape[0]
            ward_distance.append(self.ward_dist(n_left, n_right, c_left, c_right))

        ward_distance = np.array(ward_distance)
        idx = np.argmax(ward_distance)
        max_ward = ward_distance[idx]
        xmax = xgrid[idx]

        flag = self._density_est(xpca)

        return xpca, xmax, max_ward, flag

    def _density_est(self, xpca):
        """
        input: 
            xpca cluster projected onto its first principal component
        output:
            True/False flag, if x has local minima True, False o/w            
        """
        if xpca.shape[0] < 3:
            return False

        kde = KernelDensity()
        h = np.std(xpca) * (4 / 3 / len(xpca)) ** (1 / 5)
        kde.set_params(bandwidth=h).fit(xpca)

        mmin, mmax = np.percentile(xpca, [20, 80])
        xdensity = np.linspace(mmin, mmax, 1000)[:, np.newaxis]  # take .1, .9 quantile
        ydensity = np.exp(kde.score_samples(xdensity))

        local_minimas_idx = argrelmin(ydensity)[0]

        if local_minimas_idx.size == 0:
            flag = False
        else:
            flag = True

        return flag

    def _is_stop(self, ):
        """
        output:
            True if terminal criterium is met, False o/w
        """
        return np.all(self.history[:, 4] == False)

    def _which_split(self, ):
        """
        output:
            label in self.y, which cluster to split on current iteration
        """
        return np.nanargmax(self.history[:, 1])

    def ward_dist(self, n1, n2, c1, c2):
        """
        INPUT : n1, n2 - number of object in subcluster1 and 
                         subcluster2 respectively
                c1, c2 - means of subcluster1 and subcluster2 respectively, 
                         each 1 x m numpy ndarray vectors

        OUTPUT : Ward's distance between subcluster1 and subcluster2
        """
        return np.sum((c1 - c2) * (c1 - c2)) * n1 * n2 / (n1 + n2)


class CCDB():
    def __init__(self, n_clusters, random_state=42, stopping='n_clusters'):
        self.random_state = random_state
        self.n_clusters = n_clusters
        self.stopping = stopping

    def fit(self, X):

        self.X = X - X.mean(axis=0)  # z-scoring?

        self.y = np.zeros((1, X.shape[0]))
        self.max_label = 1
        self.iter = 0
        self.xfeature = []
        parent = -1

        xfeature, split_feature, xmin, ymin = self._density_est(self.X)
        self.history = np.array([[xmin, ymin, self.iter, parent, split_feature]], dtype=float)
        self.xfeature.append(xfeature)

        flag = not self._is_stop()

        while flag:
            parent = self._which_split()  # parent is an index, not data
            y = self._split(parent)
            self.y = np.concatenate((self.y, y), axis=0)

            self.iter += 1
            self.max_label = self.history.shape[0]
            flag = not self._is_stop()

            if self.y.shape[0] == self.n_clusters and self.stopping == 'n_clusters':
                flag = False

        return self.y[-1]

    def _split(self, parent):
        """
        input:
            parent label in self.y of the cluster to split
        output:
            y 1 x total_n_objects ndarray  
        """
        # init params, global y, local _y, params of the parent cluster,
        # labels for child clusters

        y = self.y[-1].copy()
        parent_feature = self.xfeature[parent]
        parent_xmin = self.history[parent, 0]
        l_label = 0 + self.max_label
        r_label = 1 + self.max_label

        # split parent cluster
        _y = np.where(parent_feature <= parent_xmin, l_label, r_label)
        y[np.where(y == parent)] = _y[:, 0]

        # precompute information for child clusters
        l_x = self.X[np.where(y == l_label)]
        r_x = self.X[np.where(y == r_label)]
        l_xfeature, l_idx, l_xmin, l_ymin = self._density_est(l_x)
        r_xfeature, r_idx, r_xmin, r_ymin = self._density_est(r_x)

        # store information for child clusters
        l_history = np.array([[l_xmin, l_ymin, self.iter, parent, l_idx]], dtype=float)
        r_history = np.array([[r_xmin, r_ymin, self.iter, parent, r_idx]], dtype=float)
        self.history = np.concatenate((self.history, l_history), axis=0)
        self.history = np.concatenate((self.history, r_history), axis=0)
        self.xfeature.append(l_xfeature)
        self.xfeature.append(r_xfeature)

        # zerod parent row since it is already divided
        self.history[parent, 0], self.history[parent, 1] = np.nan, np.nan

        return y.reshape(1, y.shape[0])

    def _density_est(self, x):
        """
        input: 
            x n_objects x n_features ndarray

        output:
            xpca cluster projected onto its first principal component
            xmin x value of minimal minima of produced kernel density
            ymin y value of minimal minima of produced kernel density
        """
        if x.shape[0] < 3: return np.nan, np.nan, np.nan, np.nan

        xidx = 0
        xfeature = []
        xmin = np.inf
        ymin = np.inf

        kde = KernelDensity()
        for i, feature in enumerate(x.T):
            h = np.std(feature) * (4 / 3 / len(feature)) ** (1 / 5)
            kde.set_params(bandwidth=h).fit(feature[:, np.newaxis])

            mmin, mmax = np.percentile(feature, [5, 95])
            xdensity = np.linspace(mmin, mmax, 1000)[:, np.newaxis]
            ydensity = np.exp(kde.score_samples(xdensity))  # think about this 1000
            # number 1000 seems pretty arbitrary for me
            local_minimas_idx = argrelmin(ydensity)[0]

            if local_minimas_idx.size != 0:
                _idx = ydensity[local_minimas_idx].argmin()
                _xmin = xdensity[local_minimas_idx[_idx]]
                _ymin = ydensity[local_minimas_idx[_idx]]

                if _ymin < ymin:
                    xfeature = feature
                    xmin = _xmin
                    ymin = _ymin
                    xidx = i

        if xmin is np.inf:
            return np.nan, np.nan, np.nan, np.nan

        return xfeature[:, np.newaxis], xidx, xmin, ymin

    def _is_stop(self, ):
        """
        output:
            True if terminal criterium is met, False o/w
        """
        return np.all(np.isnan(self.history[:, 0]))

    def _which_split(self, ):
        """
        output:
            label in self.y, which cluster to split on current iteration
        """
        return np.nanargmin(self.history[:, 1])

def main():
    n_clusters = 20
    x, y = make_blobs(n_samples=10000, n_features=20, centers=n_clusters)

    start = time.time()
    print("Start running dePDDP...")
    depddp = dePDDP()
    y_pred = depddp.fit(x)
    print(adjusted_rand_score(y, y_pred))

    # print("Success, execution time: {:.3f}\ndePDDP performance:".format(time.time() - start))
    # for _y in depddp.y:
    #     print(adjusted_rand_score(y, _y))
    #
    # start = time.time()
    # print("Start running WardPDB...")
    # ward_pdb = ward_based_pdb(n_clusters=n_clusters)
    # y_pred = ward_pdb.fit(x)
    #
    # print("Success, execution time: {:.3f}\nWardPDB performance:".format(time.time() - start))
    # for _y in ward_pdb.y:
    #     print(adjusted_rand_score(y, _y))


if "__name__" == "__main__":
    main()

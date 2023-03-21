import numpy as np
from numba import njit, float64, int64
from numba.experimental import jitclass

USE_CACHE=True

@njit([(float64[:],)], cache=USE_CACHE)
def calc_cumsum(v):
    cumsum = np.empty(len(v)+1, dtype=np.float64)
    cumsum[0]=0
    cumsum[1:] = np.cumsum(v)
    return cumsum

@njit([(float64[:],)], cache=USE_CACHE)
def calc_cumsum2(v):
    cumsum2 = np.empty(len(v)+1, dtype=np.float64)
    cumsum2[0]=0
    cumsum2[1:] = np.cumsum(np.square(v))
    return cumsum2



@njit([(float64[:], float64[:], int64, int64)], cache=USE_CACHE)
def calc_objective(cumsum, cumsum2, i, j):
    if j <= i:
        return 0.0
#            raise ValueError("j should never be larger than i")
    mu = (cumsum[j+1]-cumsum[i])/(j-i+1)
    result = cumsum2[j + 1] - cumsum2[i]
    result += (j - i + 1) * (mu * mu)
    result -= (2 * mu) * (cumsum[j + 1] - cumsum[i])
    return result


@jitclass([('cumsum', float64[:]), ('cumsum2', float64[:])])
class CumsumCalculator:
    def __init__(self, v):
        self.cumsum = calc_cumsum(v)
        self.cumsum2 = calc_cumsum2(v)

    def calc(self, i, j):
        return calc_objective(self.cumsum, self.cumsum2, i, j)


@njit(cache=USE_CACHE)
def calc_num_clusters(result):
    """Compute the number of clusters encoded in results
    Can be used on e.g. the result of _conventional_algorithm, Weber
    """
    num_clusters = 0
    curr_pos = len(result)-1
    while result[curr_pos]>=0:
        curr_pos = result[curr_pos]
        num_clusters+=1
    return num_clusters+1


@njit(cache=USE_CACHE)
def relabel_clusters(result):
    num_clusters = calc_num_clusters(result)-1

    out = np.empty_like(result)
    curr_pos = len(result)-1
    while result[curr_pos]>=0:
        out[result[curr_pos]:curr_pos+1] = num_clusters
        curr_pos = result[curr_pos]
        num_clusters-=1
    out[0:curr_pos+1] = num_clusters
    return out


@njit(cache=USE_CACHE)
def _optimal_univariate_microaggregation(x, k):
    n = len(x)
    assert k > 0
    if n//2 < k: # there can only be one cluster
        return np.zeros(n, dtype=np.int64)
    if k==1: # each node has its own cluster
       return np.arange(n)
    calculator = CumsumCalculator(x)


    back_tracks =  np.zeros(n, dtype=np.int64)
    min_vals = np.zeros(n)
    for i in range(0, k-1):
        min_vals[i] = np.inf
        back_tracks[i]=-1
    for i in range(k-1, 2*k-1):
        min_vals[i] = calculator.calc(0,i)
        back_tracks[i]=-1

    for i in range(2*k-1, n):
        #print("i", i)
        min_index = i-2*k+1
        #print("min", min_index)
        prev_min_val = min_vals[min_index] + calculator.calc(min_index+1, i)
        for j in range(i-2*k + 2, i-k+1):
            #print(j, min_vals[j], prev_min_val)
            new_val = min_vals[j] + calculator.calc(j+1, i)
            if  new_val < prev_min_val:
                min_index = j
                prev_min_val = new_val
        #print("result", min_index, prev_min_val)

        back_tracks[i] = min_index
        min_vals[i] = prev_min_val
        #print(back_tracks)
    return relabel_clusters(back_tracks)



def undo_argsort(sorted_arr, order):
    revert = np.empty_like(order)
    revert[order]=np.arange(len(sorted_arr))
    return sorted_arr[revert]



def optimal_univariate_microaggregation_1d(x, k):
    """Performs optimal 1d univariate microaggregation"""
    x = np.squeeze(np.asarray(x))
    assert len(x.shape)==1, "provided array is not 1d"
    assert k > 0, f"negative or zero values for k({k}) are not supported"
    assert k <= len(x), f"values of k({k}) larger than the length of the provided array ({len(x)}) are not supported"

    order = np.argsort(x)
    x = np.array(x, dtype=np.float64)[order]

    clusters = _optimal_univariate_microaggregation(x, k)
    return undo_argsort(clusters, order)

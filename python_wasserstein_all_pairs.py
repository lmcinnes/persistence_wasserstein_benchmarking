import numpy as np
import ot
import argparse
from sklearn.metrics import pairwise_distances
from typing import Union, Sequence, AnyStr



SQRT_2 = np.sqrt(2)

# @profile
def wasserstein_diagram_distance(
    pts0: np.ndarray, 
    pts1: np.ndarray, 
    y_axis: AnyStr = "death", 
    p: Union[int, float] = 1, 
    internal_q: int = 2
) -> float:
    """Compute the Persistant p-Wasserstein distance between the diagrams pts0, pts1

    Parameters
    ----------
    pts0: array of shape (n_top_features, 2)
        The first persistence diagram

    pts1: array of shape (n_top_features, 2)
        Thew second persistence diagram

    y_axis: optional, default="death"
        What the y-axis of the diagram represents. Should be one of
            * ``"lifetime"``
            * ``"death"``

    p: int, optional (default=1)
        The p in the p-Wasserstein distance to compute
        
    internal_q: int, optional (default=2)
        The p used for the internal minowski distance between points in diagrams.

    Returns
    -------
    distance: float
        The p-Wasserstein distance between diagrams ``pts0`` and ``pts1``
    """
    if y_axis == "lifetime": # Non functional for now!
        lifetimes0 = pts0[:, 1]
        lifetimes1 = pts1[:, 1]
    elif y_axis == "death":
        lifetimes0 = pts0[:,1] - pts0[:,0]
        lifetimes1 = pts1[:,1] - pts1[:,0]
    else:
        raise ValueError("y_axis must be 'death' or 'lifetime'")


    if np.isfinite(internal_q):
        if internal_q == 1:
            extra_dist0 = lifetimes0
            extra_dist1 = lifetimes1
            pairwise_dist = pairwise_distances(pts0, pts1, metric="l1")
        elif internal_q == 2:
            extra_dist0 = lifetimes0 / SQRT_2
            extra_dist1 = lifetimes1 / SQRT_2
            pairwise_dist = pairwise_distances(pts0, pts1, metric="l2")
        else:
            extra_dist0 = lifetimes0 * (2 **(1/internal_q - 1))
            extra_dist1 = lifetimes1 * (2 **(1/internal_q - 1))
            pairwise_dist = pairwise_distances(pts0, pts1, metric="minkowski", p=internal_q)
    else:
        extra_dist0 = (pts0[:,1]-pts0[:,0])/2
        extra_dist1 = (pts1[:,1]-pts1[:,0])/2
        pairwise_dist = pairwise_distances(pts0, pts1, metric="chebyshev")

    rows_with_zeros = np.any(pairwise_dist == 0, axis=1)
    cols_with_zeros = np.any(pairwise_dist == 0, axis=0)

    if np.sum(rows_with_zeros) == pts0.shape[0] and np.sum(cols_with_zeros) == pts1.shape[0]:
        return 0.0

    pairwise_dist = pairwise_dist[~rows_with_zeros, :][:, ~cols_with_zeros]
    extra_dist0 = extra_dist0[~rows_with_zeros]
    extra_dist1 = extra_dist1[~cols_with_zeros]

    all_pairs_ground_distance_a = np.hstack([pairwise_dist, extra_dist0[:, np.newaxis]])
    extra_row = np.zeros(all_pairs_ground_distance_a.shape[1])
    extra_row[: pairwise_dist.shape[1]] = extra_dist1
    all_pairs_ground_distance_a = np.ascontiguousarray(np.vstack([all_pairs_ground_distance_a, extra_row]))

    if p != 1:
        all_pairs_ground_distance_a = all_pairs_ground_distance_a ** p

    n0 = pairwise_dist.shape[0]
    n1 = pairwise_dist.shape[1]
    a = np.ones(n0 + 1)
    a[n0] = n1
    a /= a.sum()
    b = np.ones(n1 + 1)
    b[n1] = n0
    b /= b.sum()

    base_dist = (n0 + n1) * ot.emd2(a, b, all_pairs_ground_distance_a, processes=1, numItermax=200000)

    if p != 1:
        return np.power(base_dist, 1.0 / p)
    else:
        return base_dist

if __name__ == "__main__":
    # Note that p and q are reversed to match argument formats in HERA
    parser = argparse.ArgumentParser(description='Compute all pairs distances between many persistence diagrams')
    parser.add_argument('files', metavar='FILE', nargs='+', help="Files of persistence diagrams to process")
    parser.add_argument('-o', '--output', metavar='OUTFILE', help="Filename to write output to")
    parser.add_argument('-q', metavar='INT', default=1, type=int, help="The p to use for Wasserstein W_p distance")
    parser.add_argument('-p', metavar='INT', default=2, type=int, help="The p to use for Minkowski distance in diagram space")
    parser.add_argument('-r', '--print_results', action='store_true', help="Print results to screen")
    args = parser.parse_args()
    
    all_diagrams = [np.loadtxt(file) for file in args.files]
    result = [
        wasserstein_diagram_distance(dgm1, dgm2, p=args.q, internal_q=args.p) 
        for dgm1 in all_diagrams for dgm2 in all_diagrams
    ]
    
    if args.output:
        np.savetxt(args.output, np.array(result), fmt="%.15f")
        
    if args.print_results:
        print(result)


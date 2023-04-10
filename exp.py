"""
This module provides useful tools for calculating modal parameters from
experimental data of systems with mechanical vibration.

"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root, root_scalar
from scipy.signal import find_peaks


FLOAT_TYPE = np.float64


def _get_nearest_value(a, x, loc='lower', side='left'):
    """
    Find the nearest value to `x` in the sorted array `a`.

    Parameters
    ------
    a : ndarray
        Sorted array to search in.
    x : float
        Value to search for.
    loc : str, 'lower' or 'upper', default: 'lower'
        Determines whether the function should return the nearest value
        lower than x or the nearest value greater than x.
    side : str, 'left' or 'right'
        Determines which side of the nearest value to return.

    Returns
    ------
    nearest : float
        The nearest value to x in a.

    Raises
    ------
    ValueError : If loc is not 'lower' or 'upper'.
    ValueError : If no lower value is found and loc is set to 'lower'.
    ValueError : If no upper value is found and loc is set to 'upper'.

    """
    index = np.searchsorted(a, x, side=side)
    if loc == 'lower':
        if index < 1:
            raise ValueError('No lower value in the array')
        return a[index - 1]
    elif loc == 'upper':
        if index >= a.shape[0]:
            raise ValueError('No upper value in the array')
        return a[index]
    else:
        raise ValueError(f'loc={loc} is not a valid loc')


def _get_natfreq_unique(receptance, s, **kwargs):
    peaks_index, properties = find_peaks(receptance, **kwargs)
    natfreq = s[peaks_index]
    return natfreq, properties


def get_natfreq(receptance, s_eval, **kwargs):
    """
    Find the natural frequencies from a receptance matrix evaluated at a
    set of frequencies. The peaks used for finding the natural
    frequencies correspond to the peaks in the summed receptance matrix
    to minimize error.

    Parameters
    ------
    receptance : array_like, shape(n_eval, n, n)
        The receptance matrix evaluated at `s`.
    s_eval : array_like, shape(n_eval,)
        The set of frequencies evaluated.
    **kwargs : optional
        Keyword arguments passed to the scipy.signal.find_peaks function.

    Returns
    ------
    natfreq : ndarray, shape(n,)
        The natural frequencies.
    properties : dict
        The properties of the peaks returned by scipy.signal.find_peaks.

    """
    recept_sum = np.sum(receptance, axis=(1, 2))

    peaks_index, properties = find_peaks(recept_sum, **kwargs)
    natfreq = s_eval[peaks_index]

    return natfreq, properties


def _root_always(fun, x0, args=None, atol=1e-4, maxiter=5):
    # TODO: Documentation
    # It is assumed to always find a solution regardless if the
    # solver do not reach the tolerances in maxiter. This is
    # valid because a very good initial guess is provided.
    try:
        sol = root(fun, x0=x0, args=args,
                   method='krylov',
                   options={'fatol': 1.11e-16,
                            'xatol': atol,
                            'maxiter': maxiter})
    except Exception:
        return x0
    else:
        return sol.x


def get_damping_ratio(receptance, s, natfreq, interp='linear',
                      better_tol=True, atol=1e-4, maxiter=5):
    """
    Compute the damping ratio from the receptance matrix evaluated at a
    set of frequencies and corresponding natural frequencies.

    Parameters
    ------
    receptance : array_like, shape(n_eval, n, n)
        The receptance matrix evaluated at `s`.
    s : array_like, shape(n_eval,)
        The set of frequencies evaluated.
    natfreq : ndarray, shape(n,)
        Natural frequencies.
    interp : str, optional
        Specifies the kind of interpolation as a string. The string has
        to be one of `scipy.interpolate.interp1d` kinds.
    better_tol : bool, optional
        Use a root-finding algorithm to find a more accurate solution.
    atol : float, optional
        Absolute tolerance for the root-finding algorithm.
    maxiter : int, optional
        Maximum number of iterations for the root-finding algorithm.

    Returns
    ------
    damp_ratio : ndarray, shape(n,
        Damping ratio.

    """
    # Finding `||Hi|| / sqrt(2)` by minimizing the following function
    # Note that `20 * np.log10(np.sqrt(2)) == 10 * np.log10(2)`
    _func = lambda s, i: \
        np.abs(recept_fun(s) - (recept_natfreq[i] - 10 * np.log10(2)))

    damp_ratio = np.zeros_like(natfreq, dtype=FLOAT_TYPE)
    recept_fun = interp1d(s, receptance, kind=interp, assume_sorted=True)
    recept_natfreq = recept_fun(natfreq)

    for _ in range(natfreq.shape[0]):
        omega_indices, properties = find_peaks(- _func(s, _))
        omega_peaks = np.take(s, omega_indices)
        omega_a = _get_nearest_value(omega_peaks, natfreq[_], loc='lower')
        omega_b = _get_nearest_value(omega_peaks, natfreq[_], loc='upper')

        if better_tol:
            omega_a_better = _root_always(_func, x0=omega_a, args=(_,),
                                          atol=atol, maxiter=maxiter)
            if omega_a_better < natfreq[_]:
                omega_a = omega_a_better
            omega_b_better = _root_always(_func, x0=omega_b, args=(_,),
                                          atol=atol, maxiter=maxiter)
            if omega_b_better > natfreq[_]:
                omega_b = omega_b_better

        damp_ratio[_] = (omega_b - omega_a) / (2 * natfreq[_])

    return damp_ratio

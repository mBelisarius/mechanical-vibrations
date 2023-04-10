"""
TODO: titles and axes labels using for subplots using 'sup' are slighty off-center

"""
import math
import matplotlib.pyplot as plt
import numpy as np


FLOAT_TYPE = np.float64


def _round_to_n(x, n):
    """
    Round a number to n significant algharisms.

    Parameters
    ------
    x : float
        Number to be rounded
    n : int
        Number of significant algharisms.

    Returns
    ------
    rounded : float
        Rounded number.

    """
    _multiplier = 10 ** (math.ceil(math.log10(abs(x) + 1.)))
    return round(x / _multiplier, n) * _multiplier


def _get_ticks(data, n_ticks, n_dgt=None):
    """

    Parameters
    ------
    data : array_like, shape(n,)
    n_ticks : int
    n_dgt : int, optional

    Returns
    ------
    ticks : ndarray, shape(n_ticks,)

    """
    # TODO: Documentation
    x_min = np.min(data)
    x_max = np.max(data)

    if n_dgt is None:
        max_value = max(abs(x_min), abs(x_max))
        n_dgt = math.ceil(math.log10(max_value + 1.)) - 1

    x_span = np.linspace(x_min, x_max, n_ticks, dtype=FLOAT_TYPE)
    return np.array([_round_to_n(x, n_dgt) for x in x_span], dtype=FLOAT_TYPE)


def _solve_function(fun, t_eval):
    """

    Parameters
    ------
    fun : callable
    t_eval : array_like, shape(n,)

    Returns
    ------
    sol : ndarray, shape(n,)

    """
    # TODO: Documentation
    sol = np.array([fun(t) for t in t_eval], dtype=FLOAT_TYPE)
    return sol.T


def _plot(*data, vectorized=False, style=None, grid=True, xlabel=None,
          ylabel=None, units=('', ''), title=None, legend=None,
          legend_title=None, legend_loc='best', fprop=1.5, dpi=300):
    """

    Parameters
    ----------
    x, y : array_like or scalar
    vectorized : bool, optional, default: False
    style : dict, optional
    grid : bool, optional, default: True
    xlabel : str, optional
    ylabel : str, optional
    units : tuple of str, optional
    title : str, optional
    legend : sequence of Artists, optional
    legend_title : str, optional
    legend_loc : str or pair of floats, default: 'best'
    fprop : float, optional, default: 1.5
    dpi : int, optional, default: 300

    """
    # TODO: Documentation
    plt.figure(figsize=(int(fprop * 4), 4), dpi=dpi)
    if vectorized:
        for _ in range(data[0].shape[0]):
            plt.plot(data[0][_], data[1][_], **style)
    else:
        plt.plot(*data, **style)
    plt.margins(x=0., tight=True)
    if grid:
        plt.grid()
    plt.xlabel(f'{xlabel} ({units[0]})')
    plt.ylabel(f'{ylabel} ({units[1]})')
    plt.title(title)
    if legend is not None:
        plt.legend(legend, title=legend_title, loc=legend_loc)
    plt.tight_layout()
    plt.show()


def _plot_sub(*data, style=None, grid=True, xlabel=None, ylabel=None,
              units=('', ''), title=None, legend_label='Index', sharex=False,
              sharey=False, fprop=1.5, dpi=300):
    """

    Parameters
    ----------
    x, y : array_like or scalar
    style : dict, optional
    grid : bool, optional, default: True
    xlabel : str, optional
    ylabel : str, optional
    units : tuple of str, optional
    title : str, optional
    legend_label : str, optional, default: 'Index'
    sharex : bool, optional, default: False
    sharey : bool, optional, default: True
    fprop : float, optional, default: 1.5
    dpi : int, optional, default: 300

    """
    # TODO: Documentation
    # TODO: Generalize to nrows, ncols
    n_plots = data[1].shape[0]
    _data_t = np.repeat([data[0]], n_plots, axis=0) if sharex else data[0]
    fig, ax = plt.subplots(nrows=n_plots, sharex='all', sharey=sharey,
                           figsize=(int(fprop * 4), 4), dpi=dpi)

    for _ in range(n_plots):
        ax[_].plot(_data_t[_], data[1][_], **style)
        ax[_].margins(x=0., tight=True)
        if grid:
            ax[_].grid()
        ax[_].set_ylabel(f'{legend_label} {_}')

    fig.supxlabel(f'{xlabel} ({units[0]})')
    fig.supylabel(f'{ylabel} ({units[1]})')
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_modal_amplitudes(T, fprop=1.5, dpi=300):
    """

    Parameters
    ----------
    T : array_like, shape(n, n)
    fprop : float, optional, default: 1.5
    dpi : int, optional, default: 300

    """
    # TODO: Documentation
    style = {'linestyle': '--', 'marker': 'o', 'antialiased': True}
    xlabel = 'Physical coordinates index'
    ylabel = 'Modal amplitudes'
    title = 'Vibration modes'
    legend = range(T.shape[0])
    legend_title = 'Vibration mode'
    legend_loc = 'best'
    _plot(T, style=style, xlabel=xlabel, ylabel=ylabel, title=title,
          legend=legend, legend_title=legend_title, legend_loc=legend_loc,
          fprop=fprop, dpi=dpi)


def plot_forces(F, t_eval, label='', units=('s', 'N'), fprop=1.5, dpi=300):
    """

    Parameters
    ----------
    F : callable
    t_eval : array_like, shape(n,)
    label : str, optional
    units : tuple of str, optional, default: ('s', 'N')
    fprop : float, optional, default: 1.5
    dpi : int, optional, default: 300

    """
    # TODO: Documentation
    # TODO: Vectorized?
    data = np.array([F(t) for t in t_eval], dtype=FLOAT_TYPE)
    style = {'linestyle': '-', 'linewidth': 0.5, 'antialiased': True}
    xlabel = 'Time'
    ylabel = f'Force {label}'
    title = f'Forces {label}'
    legend = range(data.shape[1])
    legend_title = 'Coord. index'
    legend_loc = 'lower right'
    _plot(t_eval, data, style=style, xlabel=xlabel, ylabel=ylabel, units=units,
          title=title, legend=legend, legend_title=legend_title,
          legend_loc=legend_loc, fprop=fprop, dpi=dpi)


def plot_forces_sub(F, t_eval, label='', units=('s', 'N'), sharey=False,
                    fprop=1.5, dpi=300):
    """

    Parameters
    ----------
    F : callable
    t_eval : array_like, shape(n,)
    label : str, optional
    units : tuple of str, optional, default: ('s', 'N')
    sharey : bool, optional, default: True
    fprop : float, optional, default: 1.5
    dpi : int, optional, default: 300

    """
    # TODO: Documentation
    data = _solve_function(F, t_eval)

    style = {'linestyle': '-', 'linewidth': 0.5, 'antialiased': True}
    xlabel = 'Time'
    ylabel = f'Force {label}'
    title = f'Forces {label}'
    legend_label = 'Coord.'
    _plot_sub(t_eval, data, style=style, xlabel=xlabel, ylabel=ylabel,
              units=units, title=title, legend_label=legend_label, sharex=True,
              sharey=sharey, fprop=fprop, dpi=dpi)


def plot_vibr(t, x, vectorized=False, label='', units=('s', 'm'),
              fprop=1.5, dpi=300):
    """

    Parameters
    ----------
    t : array_like
    x : array_like
    vectorized : bool, optional, default: True
    label : str, optional
    units : tuple of str, optional, default: ('s', 'm')
    fprop : float, optional, default: 1.5
    dpi : int, optional, default: 300

    """
    # TODO: Documentation
    _t = np.array(t, dtype=object)
    _x = np.array(x, dtype=object)
    style = {'linestyle': '-', 'linewidth': 1.0, 'antialiased': True}
    xlabel = 'Time'
    ylabel = f'Vibration {label}'
    title = f'Vibrations in time {label}'
    legend = range(_t.shape[0])
    legend_title = 'Coord. index'
    legend_loc = 'lower right'

    _plot(_t, _x, vectorized=True, style=style, xlabel=xlabel, ylabel=ylabel,
          units=units, title=title, legend=legend, legend_title=legend_title,
          legend_loc=legend_loc, fprop=fprop, dpi=dpi)


def plot_vibr_sub(t, x, label='', units=('s', 'm'), sharey=False, fprop=1.5,
                  dpi=300):
    """

    Parameters
    ----------
    t : array_like
    x : array_like
    label : str, optional
    units : tuple of str, optional, default: ('s', 'm')
    sharey : bool, optional, default: True
    fprop : float, optional, default: 1.5
    dpi : int, optional, default: 300

    """
    # TODO: Documentation
    style = {'linestyle': '-', 'linewidth': 1.0, 'antialiased': True}
    xlabel = 'Time'
    ylabel = f'Vibration {label}'
    title = f'Vibration in time {label}'
    legend_label = 'Coord.'
    _plot_sub(t, x, style=style, xlabel=xlabel, ylabel=ylabel, units=units,
              title=title, legend_label=legend_label, sharex=False,
              sharey=sharey, fprop=fprop, dpi=dpi)


def plot_receptances(H, s_eval, units=('rad/s', 'dB re m/N'),
                     fprop=1.5, dpi=300):
    """

    Parameters
    ----------
    H : array_like, shape(n, n)
    s_eval : array_like, shape(n_eval,)
    units : tuple of str, optional, default: ('rad/s', 'dB re m/N')
    fprop : float, optional, default: 1.5
    dpi : int, optional, default: 300

    """
    # TODO: Documentation
    # TODO: Use _plot for the plot
    style = {'linestyle': '-', 'linewidth': 1.0, 'antialiased': True}
    plt.figure(figsize=(int(fprop * 4), 4), dpi=dpi)
    for _ in range(H.shape[1]):
        for __ in range(H.shape[2]):
            plt.plot(s_eval, H[:, _, __], label=f'H{_}{__}', **style)
    plt.margins(x=0., tight=True)
    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(0.9, 0.5))
    plt.xlabel(f'Frequency ({units[0]})')
    plt.ylabel(f'Receptance ({units[1]})')
    plt.title('Receptances')
    plt.tight_layout()
    plt.show()


def plot_receptances_sub(H, s_eval, units=('rad/s', 'dB re 1m/N'),
                         n_ticks=4, sharey=False,
                         fprop=1.5, dpi=300):
    """

    Parameters
    ----------
    H : array_like, shape(n, n)
    s_eval : array_like, shape(n_eval,)
    units : tuple of str, optional, default: ('rad/s', 'dB re m/N')
    n_ticks : int, optional, default: 4
    sharey : bool, optional, default: False
    fprop : float, optional, default: 1.5
    dpi : int, optional, default: 300

    """
    # TODO: Documentation
    # TODO: Use _plot_sub for the plot

    style = {'linestyle': '-', 'linewidth': 1.0, 'antialiased': True}
    xlabel = 'Frequency'
    ylabel = 'Receptance'
    title = 'Receptances'
    legend_label = 'H'

    ndof = H.shape[1]
    fig, ax = plt.subplots(nrows=ndof, ncols=ndof, sharex='all', sharey=sharey,
                           figsize=(int(fprop * 4), 4), dpi=dpi)

    xticks = _get_ticks(s_eval, n_ticks=n_ticks)
    if sharey:
        yticks = _get_ticks(H, n_ticks=n_ticks)

    for _ in range(ndof):
        for __ in range(ndof):

            if not sharey:
                yticks = _get_ticks(H[:, _, __], n_ticks=n_ticks)

            ax[_, __].plot(s_eval, H[:, _, __], **style)
            ax[_, __].grid()
            ax[_, __].margins(x=0., tight=True)
            ax[_, __].set_xticks(xticks)
            ax[_, __].set_yticks(yticks)
            ax[_, __].set_ylabel(f'{legend_label}{_}{__}')

    fig.supxlabel(f'{xlabel} ({units[0]})')
    fig.supylabel(f'{ylabel} ({units[1]})')
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_phases(H_phases, s_eval, units=('rad/s', 'rad'), fprop=1.5, dpi=300):
    """

    Parameters
    ----------
    H_phases : array_like, shape(n, n)
    s_eval : array_like, shape(n_eval,)
    units : tuple of str, optional, default: ('rad/s', 'dB re m/N')
    fprop : float, optional, default: 1.5
    dpi : int, optional, default: 300

    """
    # TODO: Documentation
    # TODO: Use _plot for the plot
    style = {'linestyle': '-', 'linewidth': 1.0, 'antialiased': True}
    plt.figure(figsize=(int(fprop * 4), 4), dpi=dpi)
    for _ in range(H_phases.shape[1]):
        for __ in range(H_phases.shape[2]):
            plt.plot(s_eval, H_phases[:, _, __], label=f'H{_}{__}', **style)
    plt.margins(x=0., tight=True)
    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(0.9, 0.5))
    plt.xlabel(f'Frequency ({units[0]})')
    plt.ylabel(f'Phase ({units[1]})')
    plt.title('Phases')
    plt.tight_layout()
    plt.show()


def plot_phases_sub(H_phases, s_eval, units=('rad/s', 'degree'),
                    n_ticks=4, sharey=True,
                    fprop=1.5, dpi=300):
    """

    Parameters
    ----------
    H_phases : array_like, shape(n, n)
    s_eval : array_like, shape(n_eval,)
    units : tuple of str, optional, default: ('rad/s', 'dB re m/N')
    n_ticks : int, optional, default: 4
    sharey : bool, optional, default: False
    fprop : float, optional, default: 1.5
    dpi : int, optional, default: 300

    """
    # TODO: Documentation
    # TODO: Use _plot_sub for the plot
    style = {'linestyle': '-', 'linewidth': 1.0, 'antialiased': True}
    xlabel = 'Frequency'
    ylabel = 'Phase'
    title = 'Phases'
    legend_label = 'phi'

    ndof = H_phases.shape[1]
    fig, ax = plt.subplots(nrows=ndof, ncols=ndof, sharex='all', sharey=sharey,
                           figsize=(int(fprop * 4), 4), dpi=dpi)

    xticks = _get_ticks(s_eval, n_ticks=n_ticks)
    if sharey:
        yticks = _get_ticks(H_phases, n_ticks=n_ticks)

    for _ in range(ndof):
        for __ in range(ndof):

            if not sharey:
                yticks = _get_ticks(H_phases[:, _, __], n_ticks=n_ticks)

            ax[_, __].plot(s_eval, H_phases[:, _, __], **style)
            ax[_, __].grid()
            ax[_, __].margins(x=0., tight=True)
            ax[_, __].set_xticks(xticks)
            ax[_, __].set_yticks(yticks)
            ax[_, __].set_ylabel(f'{legend_label}{_}{__}')

    fig.supxlabel(f'{xlabel} ({units[0]})')
    fig.supylabel(f'{ylabel} ({units[1]})')
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()

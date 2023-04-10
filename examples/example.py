import numpy as np
from vibr import VibrSystem
from exp import get_natfreq, get_damping_ratio
from plotutils import \
    plot_modal_amplitudes, \
    plot_forces, \
    plot_forces_sub, \
    plot_vibr, plot_vibr_sub, \
    plot_receptances, \
    plot_receptances_sub, \
    plot_phases, \
    plot_phases_sub


if __name__ == '__main__':
    m1, m2, m3 = 8760, 1752, 350
    k1, k2, k3 = 8.76e5, 8.76e4, 3.50e5
    alpha, beta = 0., 0.005

    M = np.array([[m1, 0, 0], [0, m2, 0], [0, 0, m3]], dtype=np.float64)
    K = np.array([[k1 + k2, -k2, 0], [-k2, k2 + k3, -k3], [0, -k3, k3]],
                 dtype=np.float64)

    vibr_sys = VibrSystem(M, K)
    vibr_sys.uncouple_linear()
    vibr_sys.set_ab_damping(alpha, beta)

    F = lambda t: [1000 * np.sin(vibr_sys.get_natfreq()[2] * t), 0, 0]
    x0 = np.array([[0, 0], [0, 0], [0, 0]], dtype=np.float64)

    vibr_sys.set_forces(F)
    sol_t, sol_r, sol_y = vibr_sys.solve_uncoupled(x0, t_span=(0, 10))

    s_eval = np.arange(0, 70, 1e-2, dtype=np.float64)
    vibr_sys.solve_receptances(s_eval)

    t_eval = np.arange(0, 10, 0.01, dtype=np.float64)

    print(f'Natural frequencies in rad/s: {vibr_sys.get_natfreq()}')
    plot_modal_amplitudes(vibr_sys.modal_matrix)
    plot_forces(F, t_eval, label='(base coordinates)')
    plot_forces_sub(F, t_eval, label='(base coordinates)')
    plot_forces(vibr_sys.modal_forces, t_eval, label='(modal coordinates)')
    plot_forces_sub(vibr_sys.modal_forces, t_eval, label='(modal coordinates)')
    plot_vibr(sol_t, sol_r, vectorized=True, label='(modal coordinates)')
    plot_vibr_sub(sol_t, sol_r, label='(modal coordinates)')
    plot_vibr(sol_t, sol_y, vectorized=True, label='(mass cordinates)')
    plot_vibr_sub(sol_t, sol_y, label='(mass cordinates)')
    plot_receptances(vibr_sys.get_recept_db(), s_eval)
    plot_receptances_sub(vibr_sys.get_recept_db(), s_eval, sharey=False, fprop=2)
    plot_phases(vibr_sys.get_recept_phases(), s_eval)
    plot_phases_sub(vibr_sys.get_recept_phases(), s_eval)

    natfreq_exp, properties_exp = get_natfreq(vibr_sys.get_recept_db(), s_eval, prominence=5, wlen=len(s_eval) // 12, distance=len(s_eval) // 1200)
    z_sum = get_damping_ratio(np.sum(vibr_sys.get_recept_db(), axis=(1, 2)), s_eval, natfreq_exp)
    z_22 = get_damping_ratio(vibr_sys.get_recept_db()[:, 2, 2], s_eval, natfreq_exp)

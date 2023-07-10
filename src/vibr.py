"""
Defines a n-DOF mechanical vibration system and methods for solving it.

"""

import itertools
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import cholesky, eig, eigh_tridiagonal, inv


class VibrSystem:
    """
    Define a vibrational system with multiple DOF.

    Parameters
    ------
    mass : array_like, shape (n, n)
        Mass matrix.
    stiff : array_like, shape (n, n)
        Stiffness matrix.
    damp : array_like, shape (n, n), optional
        Damping matrix.

    """

    def __init__(self, mass, stiff, damp=None):
        self.mass = mass
        self.stiff = stiff
        self.damp_ratio = damp

        self.forces = None
        self.damp_matrix = None
        self.spectral_matrix = None
        self.modal_matrix = None
        self.receptance = None
        self.receptance_db = None
        self.receptance_phases = None

    def set_forces(self, F):
        """
        Set the external forces function for the mechanical system.

        Parameters
        ------
        F : callable
            External forces acting on the system. The calling signature
            is F(t) and should return an array_like of shape (n,).

        """
        self.forces = F

    def set_damping_matrix(self, damp_ratio=None):
        """
        Set the damping matrix either by the proportional mass-stiffness
        method or direct input of the damping ratios.

        Parameters
        ------
        damp_ratio : array_like, shape (n,) or None, optional
            Damping ratios.

        Returns
        ------
        damp_matrix : ndarray, shape (n, n)
            Damping matrix.

        Raises
        ------
        ValueError

        """
        if damp_ratio is None and self.damp_ratio is not None:
            if self.check_uncoupled():
                damp_ratio = self.damp_ratio
            else:
                raise UserWarning('The system must be uncoupled  to use the '
                                  'proportional mass-stiffnes method')
        else:
            raise ValueError('The damping ratio must be defined')

        a_tr = np.array([[1, 1], self.spectral_matrix.diagonal()])
        b = 2 * np.outer(damp_ratio, self.get_natfreq()).diagonal()
        zeta = np.linalg.solve(a_tr.T, b)
        self.damp_matrix = zeta[0] * self.mass + zeta[1] * self.stiff
        return self.damp_matrix

    def set_ab_damping(self, alpha, beta, natfreq=None):
        """
        Set the damping ratio through the proportional mass-stiffness
        (alpha-beta) method.
            C = alpha * M + beta * K
        The natural frequencies must be know either by uncoupling the
        system or by experimentation data.

        Parameters
        ------
        alpha : float
            Mass proportion factor.
        beta : float
            Stiffness proportion factor.
        natfreq : array_like, shape (n,), optional
            Natural frequencies of the system.

        Returns
        ------
        damp_ratio : ndarray, shape (n,)
            Modal damping ratio for all the n sub-systems.
        damp_matrix : ndarray, shape (n, n)
            Damping matrix.

        Raises
        ------
        ValueError

        """
        if natfreq is not None:
            omega = np.array(natfreq)
        elif self.check_uncoupled():
            omega = self.get_natfreq()
        else:
            raise ValueError('The system must be uncoupled first to set the '
                             'damping through the alpha-beta (proportional '
                             'mass-stiffness) method, or the natural '
                             'frequencies must be passed through the natfreq '
                             'parameter')

        self.damp_ratio = (alpha / (2 * omega)) + ((beta * omega) / 2)
        self.damp_matrix = alpha * self.mass + beta * self.stiff
        return self.damp_ratio, self.damp_matrix

    def get_natfreq(self):
        """
        Get the natural frequencies through the spectral matrix.

        Retuns
        ------
        natfreq : ndarray, shape (n,)
            Natural frequencies of the system.

        Raises
        ------
        ValueError

        """
        if not self.check_uncoupled():
            raise ValueError('The system must be uncoupled first')

        if isinstance(self.spectral_matrix, np.ndarray):
            return np.sqrt(self.spectral_matrix.diagonal())
        else:
            return np.sqrt(self.spectral_matrix)

    @staticmethod
    def vibr_ode(t, y, m, c, k, f):
        """
        Second order linear ordinary differential equation that models a
        1-DOF mechanical system in vibration according to Newton's
        second law.

        Parameters
        ------
        t : float
            Time at which the system's position is evaluated.
        y : array_like, shape (2,)
            Initial value for position and velocity.
        m : float
            Mass.
        c : float
            Damping coefficient.
        k : float
            Stiffness coefficient.
        f : callable
            External forces function.

        Returns
        ------
        pos : float
            Position of the system at time t.
        vel : float
            Velocity of the system at time t.

        """
        return y[1], (-c * y[1] - k * y[0] + f(t)) / m

    @staticmethod
    def solve_vibr(m, c, k, f, y0, t_span, **options):
        """
        Solve an 1-DOF mechanical system in vibration modeled as a
        second order linear ordinary differential equation.

        Paraneters
        ------
        m : float
            Mass.
        c : float
            Damping coefficient.
        k : float
            Stiffness coefficient.
        f : callable
            External forces function.
        y0 : array_like, shape(2,)
            Initial conditions for position and velocity.
        t_span : 2-member sequence
            Interval of integration (t0, tf).
        **options
            Options passed to scipy.integrate.solve_ivp.

        Returns
        ------
        sol_t : ndarray, shape (n_points,)
            Time points.
        sol_y : ndarray, shape (n_points,)
            Values of the position at time t.

        Raises
        ------
        ValueError

        """
        sol = solve_ivp(VibrSystem.vibr_ode, t_span, y0,
                        args=(m, c, k, f),
                        method='Radau', **options)

        if sol.success:
            return sol.t, sol.y[0]
        else:
            raise ValueError(sol.message)

    def uncouple(self):
        """
        Uncouple a n-DOF vibrational system into n 1-DOF systems by
        solving an eigenvalue problem.

        Returns
        ------
        S : ndarray, shape(n, n)
            Spectral matrix.
        T : ndarray, shape(n, n)
            Modal matrix.

        """
        # Choleski decomposition
        mass_upper = cholesky(self.mass, lower=False)

        # Mass normalized stiffness matrix
        mass_upper_inv = inv(mass_upper)
        mass_upper_tr_inv = inv(mass_upper.T)
        stiff_norm = mass_upper_tr_inv @ self.stiff @ mass_upper_inv

        # Eigenvalue problem
        sol_eig = eig(stiff_norm)
        w = sol_eig[0]
        v = sol_eig[1:][0]
        idx = w.argsort(kind='stable')
        w = w[idx]
        v = v[:, idx]

        self.spectral_matrix = np.diag(w)
        self.modal_matrix = mass_upper_inv @ v
        return self.spectral_matrix, self.modal_matrix

    def uncouple_linear(self):
        """
        Uncouple a n-DOF vibrational system into n 1-DOF systems by
        solving an eigenvalue problem.
        Works only for linear systems and provide a 20x faster
        computation by using linearity properties for the decomposition.

        Returns
        ------
        S : ndarray, shape(n, n)
            Spectral matrix.
        T : ndarray, shape(n, n)
            Modal matrix.

        """
        if not np.array_equal(self.mass, np.diag(np.diag(self.mass))):
            raise ValueError('M is not a diagonal matrix (does not correspond '
                             'to a linear system)')

        # Choleski decomposition
        mass_upper = np.sqrt(self.mass)

        # Mass normalized stiffness
        mass_upper_inv = inv(mass_upper)
        stiff_norm = mass_upper_inv @ self.stiff @ mass_upper_inv

        # Eigenvalue problem
        w, v = eigh_tridiagonal(stiff_norm.diagonal(), stiff_norm.diagonal(-1))

        self.spectral_matrix = np.diag(w)
        self.modal_matrix = mass_upper_inv @ v
        return self.spectral_matrix, self.modal_matrix

    def check_uncoupled(self):
        """
        Checks if the system is uncoupled.

        Returns
        -------
        is_uncoupled : bool
            Returns True if the system was already uncoupled.

        """
        return self.spectral_matrix is not None

    def modal_forces(self, t):
        """
        Evaluate the modal external forces at time t. Note that the
        system must be uncoupled first.

        Parameters
        ------
        t : float
            Time which the external forces function is evaluated.

        Returns
        ------
        modal_forces : ndarray, shape (n,)
            Modal external forces evalueated at time t.

        """
        return self.modal_matrix.T @ self.forces(t)

    def solve_uncoupled(self, y0, t_span, **options):
        """
        Solve the n-DOF uncoupled mechanical vibration system equations.

        Parameters
        ------
        y0 : array_like, shape (n,)
            Initial state (position and velocity) for the n sub-systems.
            Should be given as [[x0, v0], [x1, v1], ... [xn, vn]].
        t_span : 2-member sequence
            Interval of integration (t0, tf).
        **options
            Options passed to scipy.integrate.solve_ivp.

        Returns
        ------
        sol_t : list, shape (n,)
            List of the times at which the solution was evaluated for
            each uncoupled sub-system.
        sol_r : list, shape (n,)
            List of vibration amplitudes in modal coordinates evaluated
            at time t for each uncoupled sub-system.
        sol_y : list, shape (n,)
            List of vibration amplitudes in mass-reference coordinates
            evaluated at time t for each uncoupled sub-system.

        """
        if not self.check_uncoupled():
            raise ValueError('The system must be uncoupled first')

        # TODO: convert lists to ndarray
        _n_dof = len(self.spectral_matrix)
        sol_t = _n_dof * [[]]
        sol_r = _n_dof * [[]]
        sol_y = _n_dof * [0]

        t_inv = inv(self.modal_matrix)
        r0 = t_inv @ y0
        f_r = lambda t: self.modal_matrix.T[_] @ self.forces(t)

        for _ in range(_n_dof):
            k = self.spectral_matrix[_, _]
            c = 2 * self.damp_ratio[_] * np.sqrt(k)
            sol_t[_], sol_r[_] = self.solve_vibr(1., c, k, f_r, r0[_],
                                                 t_span, **options)

        # sol_y = T @ sol_r
        # TODO: use @ operator
        for _, __ in itertools.product(range(_n_dof), range(len(sol_r))):
            sol_y[_] += self.modal_matrix[_, __] * sol_r[_]

        sol_t = np.array(sol_t, dtype=object)
        sol_r = np.array(sol_r, dtype=object)
        sol_y = np.array(sol_y, dtype=object)
        return sol_t, sol_r, sol_y

    def solve_receptances(self, s_eval):
        """
        Evaluate the receptance matrices at frequencies s_eval.

        Parameters
        ------
        s_eval : array_like, shape (n_points,)
            Frequencies at which the receptances are evaluated.

        Returns
        ------
        receptances : ndarray, shape (n_points, n, n)
            Sequence of receptance matrices evaluated at s_eval.

        """
        self.receptance = np.array(
            [inv(self.stiff - self.mass * s ** 2 + 1j * self.damp_matrix * s)
             for s in s_eval])
        return self.receptance

    def get_recept_db(self):
        """
        Get the receptance matrices in dB scale.

        Returns
        -------
        receptances_db : ndarray, shape (n_points, n, n)
            Sequence of receptance matrices in dB scale evaluated at
            s_eval.

        Raises
        -------
        ValueError

        """
        if self.receptance is None:
            raise ValueError('The receptances matrices are not defined.')

        self.receptance_db = 20 * np.log10(np.abs(self.receptance))
        return self.receptance_db

    def get_recept_phases(self):
        """
        Get the phases of the receptance matrices.

        Returns
        -------
        receptances_phases : ndarray, shape (n_points, n, n)
            Sequence of phases of the receptance matrices evaluated at
            s_eval.

        Raises
        -------
        ValueError

        """
        # TODO: Validate results
        if self.receptance is None:
            raise ValueError('The receptances matrices are not defined.')

        self.receptance_phases = np.arctan2(np.imag(self.receptance),
                                            np.real(self.receptance))
        return self.receptance_phases

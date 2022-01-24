from integrators import get_integrator
import numpy as np


class Admittance_Controller:
    def __init__(self, m, b, k, r, traj):
        """System parameters"""
        self.m = m  # virtual mass [kg]
        self.b = b  # virtual damping coefficient [Ns/m]
        self.k = k  # virtual stiffness [N/m]
        self.r = r  # arm length [m]

        """Trajectory to follow"""
        self.traj = traj  # [rad]

        """Time"""
        start_time = 0  # [s]
        end_time = 1000  # [s]
        self.num_timesteps = 5000
        self.dt = (end_time - start_time) / self.num_timesteps  # timestep [s]
        self.t = np.linspace(start_time, end_time, self.num_timesteps)  # time [s]

        """Iterator"""
        self.p = 0

        """Integrator"""
        self.integrator = get_integrator(self.dt, self.q_dot)

    """Update"""

    def __call__(self, q, F_ext):
        self.F_ext = F_ext
        q = np.array(q).flatten()
        q = self.integrator.step(self.t[self.p], q)
        # self.q = odeint(self.q_dot, self.q_0.flatten(), self.t[self.p+1])
        self.p += 1

        """Send theta_d to position controller"""
        return q[0]

    """Derivative of state"""

    def q_dot(self, t, q):
        p = self.p

        """Discrete trajectory"""
        theta_0 = self.traj[p]
        if self.p == 0:
            theta_0_dot = (self.traj[p + 1] - self.traj[p]) / (2 * self.dt)
            theta_0_ddot = (
                self.traj[p + 1] - 2 * self.traj[p] + self.traj[p]
            ) / self.dt ** 2
        elif self.p == self.num_timesteps - 1:
            theta_0_dot = (self.traj[p] - self.traj[p - 1]) / (2 * self.dt)
            theta_0_ddot = (
                self.traj[p] - 2 * self.traj[p] + self.traj[p - 1]
            ) / self.dt ** 2
        else:
            theta_0_dot = (self.traj[p + 1] - self.traj[p - 1]) / (2 * self.dt)
            theta_0_ddot = (
                self.traj[p + 1] - 2 * self.traj[p] + self.traj[p - 1]
            ) / self.dt ** 2

        theta_d = q[0]
        theta_d_dot = q[1]

        """
		Admittance control and error equations:
		e'' = (1/M)(F_ext/r - b*e' - k*e)
		e = theta_d - theta_0
		"""
        theta_d_ddot = (1 / self.m) * (
            (self.F_ext / self.r)
            - self.b * (theta_d_dot - theta_0_dot)
            - self.k * (theta_d - theta_0)
        ) + theta_0_ddot

        q_dot = np.array([[theta_d_dot], [theta_d_ddot]])

        return q_dot.flatten()

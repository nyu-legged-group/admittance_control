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
        # start_time = 0  # [s]
        # end_time = 1000  # [s]
        self.num_timesteps = 5000
        # self.dt = (end_time - start_time) / self.num_timesteps  # timestep [s]
        self.dt = 1e-3
        # self.t = np.linspace(start_time, end_time, self.num_timesteps)  # time [s]

        """Iterator"""
        self.p = 0

        """Desired state"""
        self.theta_d = traj[0]
        self.theta_d_dot = 0

        """Integrator"""
        # self.integrator = get_integrator(self.dt, self.q_dot)

    """Update"""

    def __call__(self, F_ext):
        self.F_ext = F_ext
        q = np.array([[self.theta_d], [self.theta_d_dot]])
        q_dot = self.q_dot(q)
        theta_d_ddot = q_dot[1]

        """Step"""
        self.theta_d_dot += theta_d_ddot * self.dt
        self.theta_d += self.theta_d_dot * self.dt + 0.5 * theta_d_ddot * self.dt ** 2

        # q_d = self.integrator.step(t, q_d.flatten())

        # self.q = odeint(self.q_dot, self.q_0.flatten(), self.t[self.p+1])
        self.p += 1
        # theta_d = q_d[0]

        """Send theta_d to position controller"""
        return self.theta_d

    """Derivative of state"""

    def q_dot(self, q):
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

        """
        theta_d_ddot = (
            self.F_ext / (self.m * self.r * np.cos(theta_d))
            - (self.b / self.m)
            * (theta_d_dot - (np.cos(theta_0) / np.cos(theta_d)) * theta_0_dot)
            - (self.k / self.m)
            * (np.tan(theta_d) - (np.sin(theta_0) / np.cos(theta_d)))
            + np.tan(theta_d) * theta_d_dot ** 2
            + (np.cos(theta_0) / np.cos(theta_d)) * theta_0_ddot
            - (np.sin(theta_0) / np.cos(theta_d)) * theta_0_dot ** 2
        )
        """

        q_dot = np.array([[theta_d_dot], [theta_d_ddot]])

        return q_dot.flatten()

def get_integrator(dt, model, integrator="RK4"):
    """Factory for integrators: Euler, Heun, RK4, AB2"""
    integrators = dict(
        Euler=Euler(dt, model),
        Heun=Heun(dt, model),
        RK4=RungeKutta4(dt, model),
        AB2=AdamsBashforth2(dt, model)
        )    
    return integrators[integrator]

class Integrator:
    """Integrator for a system of first-order ordinary differential equations
    of the form \dot x = f(t, x, u).
    """
    def __init__(self, dt, f):
        self.dt = dt
        self.f = f

    def step(self, t, x, u):
        raise NotImplementedError

class Euler(Integrator):
    def step(self, t, x, u):
        return x + self.dt * self.f(t, x, u)

class Heun(Integrator):
    def step(self, t, x, u):
        intg = Euler(self.dt, self.f)
        xe = intg.step(t, x, u) # Euler predictor step
        return x + 0.5*self.dt * (self.f(t, x, u) + self.f(t+self.dt, xe, u))

class RungeKutta4(Integrator):
    def step(self, t, x):
        k1 = self.f(t, x)
        k2 = self.f(t+0.5*self.dt, x+0.5*self.dt*k1)
        k3 = self.f(t+0.5*self.dt, x+0.5*self.dt*k2)
        k4 = self.f(t+    self.dt, x+    self.dt*k3)
        return x + self.dt * (k1 + 2*k2 + 2*k3 + k4) / 6

class AdamsBashforth2(Integrator):
    def __init__(self, dt, f):
        super().__init__(dt, f)
        self.first_time_step = True
        self.k1 = 0

    def step(self, t, x, u):
        if self.first_time_step:
            self.k1 = self.f(t, x, u)
            intg = RungeKutta4(self.dt, self.f)
            x = intg.step(t, x, u) # RungeKutta4 step
            self.first_time_step = False
        else:
            k2 = self.f(t, x, u)
            x = x + 0.5*self.dt * (3*k2 - self.k1)
            self.k1 = k2
        return x

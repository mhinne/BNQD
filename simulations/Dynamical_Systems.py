import numpy as np
import matplotlib.pyplot as plt

class DynamicalSystemDiscontinuity():
    def __init__(self, D=1):
        """
        Dynamical system discontinuity design abstract class
        @param D (int) number of coupled equations
        """
        self.D = D

    def simulate_rk45(self, init_pos, x_start, x_end, h=1e-5):
        """
        Simulate system with Runge Kutta 4-5 from x_start to x_end
        with step size h with a discontinutiy at timepoint.

        @param init_pos (array) initial conditions of system
        @param x_start (float) start point of simulation
        @param x_end (float) ending point of simulation
        @param h (float) size of the timesteps

        @return: Y (array) simulated timeseries of length N
        """
        N = int((x_end - x_start) / h)
        Y = np.zeros((N, self.D))
        Y[0,:] = init_pos

        for t in range(N - 1):
            k1 = self.f(t, Y[t,:], h)
            k2 = self.f(t+h/2, Y[t,:]+h/2*k1, h)
            k3 = self.f(t+h/2, Y[t,:]+h/2*k2, h)
            k4 = self.f(t + h, Y[t,:]+h*k3, h)
            Y[t + 1,:] = Y[t,:] + h * (k1 + 2*k2 + 2*k3 + k4) / 6
        return Y

    def get_random_samples(self, Y, N, x_start, x_end, h):
        """
        Get samples from the system at random input locations.
        For now assumes the same input locations on all dimensions
        @param N (int) number of samples to draw from
        @return:
        """
        idx = np.arange(Y.shape[0])
        sample_idx = np.sort(np.random.choice(idx, N))

        # Get sampled irregular time indexes
        Y = Y[sample_idx,:]
        X = np.zeros((Y.shape))
        for i in range(Y.shape[1]):
            X[:,i] = sample_idx * h
        return X, Y

    def f(self, t, Y, h):
        """
        Dynamical System transition function
        """
        pass

    def plot_timeseries(self, X_time_series, time_series, x0, X, Y):
        """
        Plot time series of dynamical system
        @param X (array of [N,D]) input locations
        @param Y (array of [N,D]) time series
        @param x0 (float) discontinuity location
        @param X ()
        @param Y ()
        """
        N, D = Y.shape

        fig, ax = plt.subplots(1, 1, figsize=(12,6))
        for i in range(D):
            ax.plot(X_time_series, time_series[:,i])
            ax.scatter(X[:,i], Y[:,i], color='black', marker='x')
        ax.axvline(x0, linestyle='-', color='gray', label='Intervention threshold')
        ax.set_xlim((X[0,0], X[-1,0]))
        ax.legend()
        return fig, ax


class LotkaVolterra(DynamicalSystemDiscontinuity):
    """
    Lotka Volterra dynamical system with a discontinuity in the alpha parameter (the meek shall inherit..)
    """
    def __init__(self, alpha=1.1, beta=0.4, delta=0.1, gamma=0.4, x0=0, alpha_disc=2.1):
        super().__init__(D=2)
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma
        self.x0 = x0
        self.alpha_disc = alpha_disc

    def f(self, t, Y, h):
        x, y = Y
        alpha = self.alpha if t<self.x0/h else self.alpha_disc
        return np.array([alpha*x-self.beta*x*y, self.delta*x*y-self.gamma*y])


class MackeyGlass(DynamicalSystemDiscontinuity):
    def __init__(self, a, b, tau, n_power, x0, tau_disc):
        """
        Mackey Glass dynamical system simulation
        @param a:
        @param b:
        @param c:
        @param tau:
        @param x0 (float) discontinuity input location
        """
        super().__init__(D=1)
        self.a = a
        self.b = b
        self.tau = tau
        self.n_power = n_power
        self.x0 = x0
        self.tau_disc = tau_disc

    def f(self, t, Y, Y_delay, h):
        """
        Mackey-glass transition functio
        @param t: time index
        @param Y: array of all simulated values so far
        @return:
        """
        return self.a*Y_delay/(1+Y_delay**self.n_power) - self.b*Y

    def simulate_rk45(self, init_pos, x_start, x_end, h=1e-5):
        """
        Simulate system with Runge Kutta 4-5 from x_start to x_end
        with step size h with a discontinutiy at timepoint.

        @param init_pos (array) initial conditions of system
        @param x_start (float) start point of simulation
        @param x_end (float) ending point of simulation
        @param h (float) size of the timesteps

        @return: Y (array) simulated timeseries of length N
        """
        N = int((x_end - x_start) / h)
        Y = np.zeros((N, self.D))
        Y[0, :] = init_pos

        for t in range(N - 1):
            if t%100:
                print(t,'/',N-1)
            tau = self.tau if t < self.x0 / h else self.tau_disc
            tau_scaled = int(tau/h)
            if t > tau:
                Y_delay = Y[t - tau_scaled]
            else:
                Y_delay = 0

            k1 = self.f(t, Y[t,:], Y_delay, h)
            k2 = self.f(t + h / 2, Y[t,:] + h / 2 * k1, Y_delay + h/2, h)
            k3 = self.f(t + h / 2, Y[t,:] + h / 2 * k2, Y_delay + h/2, h)
            k4 = self.f(t + h, Y[t,:] + h * k3, Y_delay+h*k3, h)
            Y[t + 1, :] = Y[t,:] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return Y
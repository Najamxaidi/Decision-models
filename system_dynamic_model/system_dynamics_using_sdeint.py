import numpy as np
import sdeint
import matplotlib.pyplot as plt
from scipy.integrate import simps
from numpy import trapz
from scipy.optimize import basinhopping


class SystemDynamicsWithSdeint:
    """
    This class replicates the model in Dussutour (2009)
    This follows an ODE model, see equation 3.2 in the model
    """

    def __init__(self, number_of_agents, k, alpha, utility_of_choices, initial_experiences,
                 discount_rate, sd, rotation_time,
                 rotation_flag, use_fun_for_utilities, frequencies,
                  phase):
        """

        :param number_of_agents: an integer with the total number of agents
        :param k: see equation 3.1
        :param alpha: see equation 3.1
        :param utility_of_choices: an array telling about how good each choice is
        :param initial_experiences: an array telling the initial experiences of choices
        :param discount_rate: a number between 0 and 1
        """

        self.number_of_agents = number_of_agents
        self.k = k
        self.alpha = alpha  # model parameter
        self.experiences_of_choices = np.array(initial_experiences)
        self.discount_rate = discount_rate  # evaporation rate

        # get the total number of options avaliable to the agents
        self.options = len(utility_of_choices)
        self.sd = sd
        self.count = 0
        self.rotation_time = rotation_time
        self.flag = rotation_flag
        self.use_fun_for_utilities = use_fun_for_utilities
        self.frequencies = frequencies
        self.phase = phase

        if use_fun_for_utilities:
            a = sine_fun(frequency=frequencies[0], t=0, fs=8000, phase=self.phase[0])
            b = 1 - a

            a1 = sine_fun(frequency=frequencies[1], t=0, fs=8000, phase=self.phase[1])
            a2 = 1 - a1

            b1 = sine_fun(frequency=frequencies[2], t=0, fs=8000, phase=self.phase[2])
            b2 = 1 - b1

            self.utility_of_choices = np.array([a1,a2,b1,b2])
        else:
            self.utility_of_choices = np.array(utility_of_choices)

        # required for plotting
        self.orbits = []
        self.orbits_utility = []
        self.orbits_for_sum_pi_ei = []
        self.aggregate_utility = []
        for i in range(self.options):
            self.orbits.append([])
            self.orbits_utility.append([])
            self.aggregate_utility.append([])

    def rotation_of_utilitities(self):
        self.utility_of_choices = np.roll(self.utility_of_choices,1)

    def rate_of_experience(self, experience, t):

        self.count += 1
        if self.count == self.rotation_time and self.flag == True:
            self.utility_of_choices = np.roll(self.utility_of_choices, 1)
            self.count = 0

        if self.use_fun_for_utilities:
            a = sine_fun(frequency=self.frequencies[0], t=self.count, fs=8000, phase=self.phase[0])
            b = 1 - a

            a1 = sine_fun(frequency=self.frequencies[1], t=self.count, fs=8000, phase=self.phase[1])
            a2 = 1 - a1

            b1 = sine_fun(frequency=self.frequencies[2], t=self.count, fs=8000, phase=self.phase[2])
            b2 = 1 - b1

            self.utility_of_choices = np.array([a1, a2, b1, b2])

        # Equation 3.1 starts (this is pi)
        # calculate the probability based upon the initial experiences
        options_probability = (np.power((experience + self.k), self.alpha)) / np.sum(np.power((experience + self.k), self.alpha))

        # Equation 3.1 ends
        pi_qi_flux = options_probability * self.utility_of_choices * self.number_of_agents

        # plotting stuff
        for i in range(self.options):
            self.aggregate_utility[i].append(self.utility_of_choices[i])
            self.orbits[i].append(options_probability[i])
            self.orbits_utility[i].append(pi_qi_flux[i])
            if experience[i] < 0:
                experience[i] = 0

        self.orbits_for_sum_pi_ei.append(np.sum(options_probability*experience))

        row_ci = self.discount_rate * experience

        experience = pi_qi_flux - row_ci

        return experience

    def noise(self, W, t):
        sd = [self.sd] * self.options
        B = np.diag(sd)
        return B

    # numpy linspace parameters are as (start value , stop value, number of values)
    # for example linspace(0, 0.001, 50) will generate 50 values between 0 and 0.001
    #--------------------------
    # this function solves the differential equation as mentioned in the paper
    def solve(self, time_vector=np.linspace(0, 10, 10000)):

        soln = sdeint.itoint(self.rate_of_experience, self.noise, self.experiences_of_choices, time_vector)
        low_values_indices = soln < 0  # Where values are low
        soln[low_values_indices] = 0

        self.plot(soln,time_vector)

    def plot(self, soln, time_vector):
        plt.figure(1)
        plt.subplot(411)
        for i in range(len(self.orbits)):
            plt.plot(range(len(self.orbits[i])), self.orbits[i], label=("choice " + str(i)))
        # plt.ylim(-1, 1)
        plt.ylabel('proportion')
        plt.title('1st: proportion of agents vs time steps -- 2nd: experience of agents vs time steps')

        plt.subplot(412)
        for i in range(self.options):
            plt.plot(time_vector, soln[:, i], label=("choice " + str(i)))
        plt.xlabel('time')
        plt.ylabel('experience')

        plt.subplot(413)
        for i in range(self.options):
            plt.plot(range(len(self.orbits_utility[i])), self.orbits_utility[i], label=("choice " + str(i)))
        plt.xlabel('time')
        plt.ylabel('utility gained')

        plt.subplot(414)
        for i in range(self.options):
            plt.plot(range(len(self.aggregate_utility[i])), self.aggregate_utility[i], label=("choice " + str(i)))
        plt.xlabel('time')
        plt.ylabel('utility')
        plt.legend()
        plt.show()

    def return_area(self, time_vector=np.linspace(0, 10, 10000)):
        soln = sdeint.itoint(self.rate_of_experience, self.noise, self.experiences_of_choices, time_vector)
        return trapz(self.orbits_for_sum_pi_ei, range(0, len(self.orbits_for_sum_pi_ei)))

    def return_average_utility(self,time_vector=np.linspace(0, 10, 10000)):
        soln = sdeint.itoint(self.rate_of_experience, self.noise, self.experiences_of_choices, time_vector)
        return np.average(np.average(self.orbits_utility,0))


def sine_fun(frequency, t, fs, phase):
    y = (np.sin(2 * np.pi * frequency * t/ fs + phase) + 1) / 2
    return y


def main():
    sysd = SystemDynamicsWithSdeint(number_of_agents=100,
                                    k=1,
                                    alpha=2,
                                    utility_of_choices=[0,0,0,0],
                                    initial_experiences=[1, 1, 1, 1],
                                    discount_rate=0.99,
                                    sd=0,
                                    rotation_time=200,
                                    rotation_flag=False,
                                    use_fun_for_utilities = True,
                                    frequencies = [50, 10, 30],
                                    phase = [0, 0, 0]
                                    )

    sysd.solve(time_vector=np.linspace(0, 1000, 500))

if __name__ == "__main__":
    main()








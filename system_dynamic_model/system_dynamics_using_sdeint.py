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
                 discount_rate = 0.0, sd = 0.01, rotation_time = 1000, flag = False):
        """

        :param number_of_agents: an integer with the total number of agents
        :param k: see equation 3.1
        :param alpha: see equation 3.1
        :param utility_of_choices: an array telling about how good each choice is
        :param initial_experiences: an array telling the initial experiences of choices
        :param discount_rate: a number between 0 and 1
        """
        assert 0 <= discount_rate <= 1, "Discount rate should be between 0 and 1"

        self.number_of_agents = number_of_agents
        self.k = k
        self.alpha = alpha  # model parameter
        self.utility_of_choices = np.array(utility_of_choices)
        self.experiences_of_choices = np.array(initial_experiences)
        self.discount_rate = discount_rate  # evaporation rate

        # get the total number of options avaliable to the agents
        self.options = len(utility_of_choices)
        self.sd = sd
        self.count = 0
        self.rotation_time = rotation_time
        self.flag = flag

        # required for plotting
        self.orbits = []
        self.orbits_utility = []
        self.orbits_for_sum_pi_ei = []
        for i in range(self.options):
            self.orbits.append([])
            self.orbits_utility.append([])

    def rotation_of_utilitities(self):
        self.utility_of_choices = np.roll(self.utility_of_choices,1)

    def rate_of_experience(self, experience, t):

        self.count += 1
        if self.count == self.rotation_time and self.flag == True:
            self.utility_of_choices = np.roll(self.utility_of_choices, 1)
            self.count = 0

        # Equation 3.1 starts (this is pi)
        # calculate the probability based upon the initial experiences
        options_probability = (np.power((experience + self.k), self.alpha)) / np.sum(np.power((experience + self.k), self.alpha))

        # Equation 3.1 ends
        pi_qi_flux = options_probability * self.utility_of_choices * self.number_of_agents

        # plotting stuff
        for i in range(self.options):
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

        ##-------Print the area under the curve over here ---------###
        #print("using composite trapezoidal rule " + '{:18.5f}'.format(trapz(self.orbits_for_sum_pi_ei, range(0, len(self.orbits_for_sum_pi_ei)))))

        plt.figure(1)
        plt.subplot(311)
        for i in range(len(self.orbits)):
            plt.plot(range(len(self.orbits[i])), self.orbits[i], label=("choice " + str(i)))
        #plt.ylim(-1, 1)
        plt.ylabel('proportion')
        plt.title('1st: proportion of agents vs time steps -- 2nd: experience of agents vs time steps')

        plt.subplot(312)
        for i in range(self.options):
            plt.plot(time_vector, soln[:, i], label=("choice " + str(i)))
        #plt.ylim(ymin=-500)
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('experience')

        plt.subplot(313)
        for i in range(self.options):
            plt.plot(range(len(self.orbits_utility[i])), self.orbits_utility[i], label=("choice " + str(i)))
        plt.xlabel('time')
        plt.ylabel('utility')
        # plt.legend(bbox_to_anchor=(1.1, 1.05))
        # plt.legend()
        plt.show()

        # plt.subplot(313)
        # plt.plot(range(len(self.orbits_for_sum_pi_ei)), self.orbits_for_sum_pi_ei, label="test")
        # plt.xlabel('time')
        # plt.ylabel('Sum(Pi * Ei)')
        # plt.legend(bbox_to_anchor=(1.1, 1.05))
        # plt.legend()
        # plt.show()

    def return_area(self, time_vector=np.linspace(0, 10, 10000)):
        soln = sdeint.itoint(self.rate_of_experience, self.noise, self.experiences_of_choices, time_vector)
        return trapz(self.orbits_for_sum_pi_ei, range(0, len(self.orbits_for_sum_pi_ei)))

    def return_average_utility(self,time_vector=np.linspace(0, 10, 10000)):
        soln = sdeint.itoint(self.rate_of_experience, self.noise, self.experiences_of_choices, time_vector)
        return np.average(np.average(self.orbits_utility,0))

def main():
    sysd = SystemDynamicsWithSdeint(number_of_agents=100,
                                    k=1,
                                    alpha=2,
                                    utility_of_choices=[0.25,0.50,0.75,1],
                                    initial_experiences=[0.001, 0.001, 0.001, 0.001],
                                    discount_rate=0.99,
                                    sd=0,
                                    rotation_time=200,
                                    flag=True)

    sysd.solve(time_vector=np.linspace(0, 1000, 500))

if __name__ == "__main__":
    main()








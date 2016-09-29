import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import simps
from numpy import trapz


class SystemDynamics:
    """
    This class replicates the model in Dussutour (2009)
    This follows an ODE model, see equation 3.2 in the model
    """

    def __init__(self, number_of_agents, k, alpha, utility_of_choices, initial_experiences,
                 discount_rate = 0.0):
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
        self.count = 0

        self.orbits = []
        self.orbits_for_pi_ei = []
        for i in range(self.options):
            self.orbits.append([])
            self.orbits_for_pi_ei.append([])

    def rate_of_experience(self, experience, t, number_of_agents, alpha, k, discount_rate, options, utility_of_choices):

        # Equation 3.1 starts (this is pi)
        # calculate the probability based upon the initial experiences

        base = np.sum(np.power((experience + k), alpha))

        # calculate probability
        options_probability = (np.power((experience + k), alpha)) / base

        # Equation 3.1 ends
        pi_qi_flux = options_probability * utility_of_choices * number_of_agents

        row_ci = discount_rate * experience

        # plotting stuff
        self.count += 1
        for i in range(options):
            self.orbits[i].append(options_probability[i])

        experience = pi_qi_flux - row_ci

        return experience

    def solve(self, time_vector=np.linspace(0, 500, 100)):

        soln = odeint(self.rate_of_experience, self.experiences_of_choices, time_vector,
                      args=(self.number_of_agents, self.alpha, self.k,self.discount_rate,
                            self.options, self.utility_of_choices))

        for i in range(self.options):
            print("option " + str(i) + " has area under the curve as:")
            print("using composite trapezoidal rule " + '{:18.5f}'.format(trapz(soln[:, i], range(0, len(soln)))))
            #print("using composite Simpson's rule " + '{:18.5f}'.format(simps(soln[:, i], range(0, len(soln)))))

        plt.figure(1)
        plt.subplot(211)
        for i in range(len(self.orbits)):
            plt.plot(range(self.count), self.orbits[i], label=("choice " + str(i)))
        plt.ylabel('proportion')
        #plt.legend()
        plt.title('1st: proportion of agents vs time steps -- 2nd: experience of agents vs time steps')

        plt.subplot(212)
        for i in range(self.options):
            plt.plot(time_vector, soln[:, i], label=("choice " + str(i)))
        plt.xlabel('time')
        plt.ylabel('experience')
        plt.show()


def main():

    sysd = SystemDynamics(number_of_agents=100, k=1, alpha=2,
                          utility_of_choices=[0.5],
                          initial_experiences=[0.10],
                          discount_rate=1)

    sysd.solve()


if __name__ == "__main__":
    main()



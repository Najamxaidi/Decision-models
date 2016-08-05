import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class SystemDynamics:
    """
    This class replicates the model in Dussutour (2009)
    This follows an ODE model, see equation 3.2 in the model
    """

    def __init__(self, number_of_agents, k, alpha, utility_of_choices, initial_experiences,
                 discount_rate = 0.0, flow_rate = 1.0):
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
        self.utility_of_choices = utility_of_choices
        self.experiences_of_choices = initial_experiences
        self.discount_rate = discount_rate  # evaporation rate
        self.flow_rate = flow_rate

        # get the total number of options avalible to the agents
        self.options = len(utility_of_choices)

        # initialising probablity to zero for all choices
        # probablities will be calculated later
        self.options_Probability = []
        for i in range(self.options):
            self.options_Probability.append(0)

        # required for plotting
        self.orbits = []
        for i in range(self.options):
            self.orbits.append([])

    def reset(self):
        self.__init__()

    # numpy linspace parameters are as (start value , stop value, number of values)
    # for example linspace(0, 0.001, 50) will generate 50 values between 0 and 0.001
    #--------------------------
    # this function solves the differential equation as mentioned in the paper
    def solve(self, time_vector=np.linspace(0, 0.001, 50), noise_standard_deviation=0.01):

        # initial rate is dependent upon the initial parameters
        # level of noise is sampled from a normal distribution
        # ----------------
        #  random.normal parameters are np.random.normal(mean, standard_deviation, number of values to be returned)
        noise= np.random.normal(0, noise_standard_deviation, self.options)
        soln = odeint(self.rate_of_experience, self.experiences_of_choices, time_vector, args=(self.flow_rate, self.alpha, self.k))


        for i in range(len(self.agents)):
            plt.plot(time_vector, self.agents[i], label=i)
        plt.rc('lines', linewidth=2.0)
        plt.legend()
        plt.show()

def rate_of_experience(experience, t):
    # calculate the probablity of selection of the choices
    # discount my current experiences of the choices

    forgetting_factor = [self.discount_rate * i for i in experience]

    # calculate the probablity of choices
    base = 0
    for i in range(len(experience)):
        base += (pow((self.k + self.experience[i]), self.alpha))

    for i in range(len(self.options_Probability)):
        self.options_Probability[i] = (pow((self.k + self.experiences_of_choices[i]), self.alpha)) / base

    proportion_of_agents_on_each_branch = [self.flow_rate * i for i in self.options_Probability]


def main():
    sysd = SystemDynamics(number_of_agents= 100, k = 0.5, alpha = 2, utility_of_choices= [0.5, 1],
                          initial_experiences= [10, 10], discount_rate=0.001, flow_rate = 0.1)
    sysd.solve()


if __name__ == "__main__":
    main()



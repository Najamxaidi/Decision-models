import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


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

    # numpy linspace parameters are as (start value , stop value, number of values)
    # for example linspace(0, 0.001, 50) will generate 50 values between 0 and 0.001
    #--------------------------
    # this function solves the differential equation as mentioned in the paper
    def solve(self, time_vector=np.linspace(0, 100, 10000), noise_standard_deviation=10):

        # initial rate is dependent upon the initial parameters
        # level of noise is sampled from a normal distribution
        # ----------------
        soln = odeint(rate_of_experience, self.experiences_of_choices, time_vector,
                      args=(self.number_of_agents, self.alpha, self.k,self.discount_rate,
                            noise_standard_deviation, self.options, self.utility_of_choices), mxstep = 500)

        for i in range(self.options):
            plt.plot(time_vector, soln[:, i], label=("choice " + str(i)))

        #plt.rc('lines', linewidth=2.0)
        plt.ylim(ymin=-2000)
        plt.xlabel('time')
        plt.ylabel('experience')
        plt.legend()
        plt.show()


def rate_of_experience(experience, t, number_of_agents, alpha, k, discount_rate,
                       noise_standard_deviation, options, utility_of_choices):

    # calculate noise
    #  random.normal parameters are np.random.normal(mean, standard_deviation, number of values to be returned)
    noise = np.array(np.random.normal(0, noise_standard_deviation, options))

    # Equation 3.1 starts (this is pi)
    # calculate the probability based upon the initial experiences

    base = np.sum(np.power((experience + k), alpha))

    # calculate probability
    options_probability = (np.power((experience + k), alpha)) / base

    # Equation 3.1 ends
    pi_qi_flux = options_probability * utility_of_choices * number_of_agents

    row_ci = discount_rate * experience

    experience = pi_qi_flux - row_ci + noise

    return experience


def main():
    sysd = SystemDynamics(number_of_agents= 100, k = 0.5, alpha = 2, utility_of_choices= [10, 20],
                          initial_experiences= [100, 10], discount_rate=0.01)
    sysd.solve()


if __name__ == "__main__":
    main()



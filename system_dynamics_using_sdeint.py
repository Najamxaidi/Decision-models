import numpy as np
import sdeint
import matplotlib.pyplot as plt


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
        for i in range(self.options):
            self.orbits.append([])

    def rotation_of_utilitities(self):
        self.utility_of_choices = np.roll(self.utility_of_choices,1)


    def rate_of_experience(self, experience, t):
        #print(experience)

        # Equation 3.1 starts (this is pi)
        # calculate the probability based upon the initial experiences
        self.count += 1
        if self.count == self.rotation_time and self.flag == True:
            self.utility_of_choices = np.roll(self.utility_of_choices, 1)

        # calculate probability
        options_probability = (np.power((experience + self.k), self.alpha)) / np.sum(np.power((experience + self.k), self.alpha))

        # Equation 3.1 ends
        pi_qi_flux = options_probability * self.utility_of_choices * self.number_of_agents

        #plotting stuff
        for i in range(self.options):
            self.orbits[i].append(options_probability[i])

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

        # initial rate is dependent upon the initial parameters
        # level of noise is sampled from a normal distribution
        # ----------------
        soln = sdeint.itoint(self.rate_of_experience, self.noise, self.experiences_of_choices, time_vector)
        #print(self.count)

        plt.figure(1)
        plt.subplot(211)
        for i in range(len(self.orbits)):
            plt.plot(range(len(self.orbits[i])), self.orbits[i], label=("choice " + str(i)))
        plt.ylim(-2, 2)
        plt.ylabel('proportion of agents')
        plt.legend(loc='lower center')

        plt.title('1st: proportion of agents vs time steps -- 2nd: experience of agents vs time steps')

        plt.subplot(212)
        for i in range(self.options):
            plt.plot(time_vector, soln[:, i], label=("choice " + str(i)))
        #plt.rc('lines', linewidth=2.0)
        #plt.ylim(ymin=-2000)
        plt.xlabel('time')
        plt.ylabel('experience')
        plt.legend(bbox_to_anchor=(1.1, 1.05))
        plt.show()


def main():
    sysd = SystemDynamicsWithSdeint(number_of_agents= 100, k = 0.5, alpha = 2, utility_of_choices= [10, 100, 20],
                          initial_experiences= [100, 20, 50], discount_rate=0.01, sd = 10, rotation_time = 2000, flag = False)

    sysd.solve(time_vector=np.linspace(0, 4000, 10000))


if __name__ == "__main__":
    main()







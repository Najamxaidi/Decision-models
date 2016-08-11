import numpy as np
import matplotlib.pyplot as plt


class Agent_Based_Decision_Model:

    """
    The code below concerns the the decision taken be a group of agents
    between multiple choices. The code follows but not completely replicates the
    equation given in the paper Dussutour (2009). An agent based model has been developed where
    multiple agents choose from the number of choices avaliable

    """

    def __init__(self, number_of_agents, k, alpha, utility_of_choices,
                 initial_experiences, discount_rate = 0.0, noise_standard_deviation=0.0):

        """

        :param number_of_agents: an integer with the total number of agents
        :param k: see equation 3.1
        :param alpha: see equation 3.1
        :param utility_of_choices: an array telling about how good each choice is
        :param discount_rate: a number between 0 and 1.
        """

        self.numberOfAgents = number_of_agents
        self.options = len(utility_of_choices)
        # for utility
        self.utility_of_choices = np.array(utility_of_choices)
        self.k = k
        self.alpha = alpha
        self.previous_exp = np.zeros((self.options,), dtype=np.float)
        self.new_exp = np.array(initial_experiences)
        self.discount = discount_rate
        self.noise_standard_deviation = noise_standard_deviation

        # initially the probablity has been set to zero
        # it will be calculated later based upon the initial experience
        self.options_Probability = np.zeros((self.options,), dtype=np.float)

        # required for plotting
        self.orbits = []
        self.exp_orbits = []

        for i in range(self.options):
            self.orbits.append([])
            self.exp_orbits.append([])

    def step(self):

        # this array will contain the choices taken by each agent in the current step.
        # the choice taken depends upon the probability of the choices. This array needs to be cleared in
        # each step to contain information only related to that step.

        agent_choices = np.zeros((self.numberOfAgents,), dtype=np.int)

        # Equation 3.1 starts
        # calculate the probability based upon the initial experiences

        base = np.sum(np.power((self.new_exp + self.k), self.alpha))

        # calculate probability
        self.options_Probability = (np.power((self.new_exp + self.k), self.alpha)) / base

        # Equation 3.1 ends

        # each agent choose an option with a probability p
        agent_choices = np.random.choice(range(self.options), self.numberOfAgents, p=self.options_Probability)

        # the array count_of_choices contains info about the number of agents who have taken such a choice
        # the array temp_utility contains information about the total utility of a particular choice
        # these arrays need to be reset to zero in every step

        count_of_choices = np.zeros((self.options,), dtype=np.int)

        # counts how many agents have taken such a decision
        for i in range(self.options):
            count_of_choices[i] = np.size(np.where(agent_choices == i))

        # calculate the total utility of the options
        temp_utility = count_of_choices * self.utility_of_choices

        # generate a noise array
        noise = np.array(np.random.normal(0, self.noise_standard_deviation, self.options))

        # previous experience is discounted
        # new experience is calculated
        # previous experience is updated
        # check equation 3.2

        self.previous_exp *= self.discount  # row * perv exp
        #self.new_exp = (self.options_Probability * temp_utility * self.numberOfAgents) + noise
        self.new_exp = (temp_utility) + noise
        self.previous_exp += self.new_exp

        # for plotting
        for i in range(self.options):
            self.orbits[i].append(count_of_choices[i])
            self.exp_orbits[i].append(self.new_exp[i])

    def rotation_of_utilitities(self):
        self.utility_of_choices = np.roll(self.utility_of_choices,1)

    def plot(self):
        # plot the graphs

        plt.figure(1)
        plt.subplot(211)
        for i in range(len(self.orbits)):
            plt.plot(range(len(self.orbits[i])), self.orbits[i], label=("choice " + str(i)))
        #plt.rc('lines', linewidth=2.0)
        plt.ylim(-10, self.numberOfAgents+10)
        plt.ylabel('number of agents')
        #plt.xlabel('time step')
        plt.legend()
        plt.title('1st: number of agents vs time steps -- 2nd: experience of agents vs time steps')

        plt.subplot(212)
        for i in range(len(self.exp_orbits)):
            plt.plot(range(len(self.exp_orbits[i])), self.exp_orbits[i], label=("choice " + str(i)))
        #plt.rc('lines', linewidth=2.0)
        plt.ylabel('experience of agents')
        plt.xlabel('time step')
        plt.legend()
        #plt.title('experience of agents vs time steps')
        plt.show()


    def run(self, steps, rotation_step, flag):
        for i in range(steps):
            if i == rotation_step and flag == True:
                self.rotation_of_utilitities()
            self.step()
        self.plot()



def main():
    steps = 100
    rotation_step = 50
    flag = True
    d = Agent_Based_Decision_Model(number_of_agents= 100, k = 0.5, alpha = 2,
                                   utility_of_choices= [10, 100, 20], initial_experiences=[100, 20, 50],
                                   discount_rate=0.01, noise_standard_deviation=0.1)

    d.run(steps,rotation_step, flag)


if __name__ == "__main__":
    main()
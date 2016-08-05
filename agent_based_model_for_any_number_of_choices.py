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
                 initial_experiences, discount_rate = 0.0,noise_standard_deviation=0.0):

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
        self.utility_of_choices = utility_of_choices
        self.k = k
        self.alpha = alpha
        self.previous_exp = [0] * self.options
        self.new_exp = initial_experiences
        self.discount = discount_rate
        self.noise_standard_deviation = noise_standard_deviation

        # initially the probablity has been set to zero
        # it will be calculated later based upon the initial experience
        self.options_Probability = [0] * self.options

        self.agent_choices = []
        self.orbits = []
        self.exp_orbits = []
        #required for plotting
        for i in range(self.options):
            self.orbits.append([])
            self.exp_orbits.append([])

    # utility of a given choice
    def returnUtility(self, choice):
            return self.utility_of_choices[choice]

    # convert utility to experience
    # PLEASE NOTE:   currently not being used
    def utilityToExperience(self, utility):
        return utility

    def step(self):
        # this array will contain the choices taken by each agent in the current step.
        # the choice taken depends upon the probablity. This array needs to be cleared in
        # each step to contain information only related to that step.
        self.agent_choices.clear()

        # Equation 3.1 starts
        # calculate the probability based upon the initial experiences
        base = 0
        for i in range(len(self.new_exp)):
            base += (pow((self.k + self.new_exp[i]), self.alpha))

        # calculate probability
        for i in range(len(self.options_Probability)):
            self.options_Probability[i] = (pow((self.k + self.new_exp[i]), self.alpha)) / base

        # Equation 3.1 ends

        # each agent choose an option with a probability p
        for i in range(self.numberOfAgents):
            self.agent_choices.append(np.random.choice(range(self.options), p=self.options_Probability))

        # update experience based upon utility
        # calculate experience from current choices
        # the array count of choices contains info about the number of agents who have taken such a choice
        # the array temp_utility contains information about the total utility of a particular choice
        # these arrays need to be reset to zero in every step

        count_of_choices = [0] * self.options
        temp_utility = [0] * self.options

        for i in range(self.numberOfAgents):
            count_of_choices[self.agent_choices[i]] += 1
            temp_utility[self.agent_choices[i]] += self.returnUtility(self.agent_choices[i])

        # new experience becomes old
        for i in range(len(self.previous_exp)):
            self.previous_exp[i] += self.new_exp[i]
            self. previous_exp[i] = self.discount * self. previous_exp[i]

        noise = np.random.normal(0, self.noise_standard_deviation, self.options)

        # check equation 3.2
        # experience for the current step is calculated
        for i in range(len(self.previous_exp)):
            self.new_exp[i] = self.options_Probability[i]*temp_utility[i]*self.numberOfAgents - self.previous_exp[i] + noise[i]

        for i in range(self.options):
            self.orbits[i].append(count_of_choices[i])
            self.exp_orbits[i].append(self.new_exp[i])

    def plot(self):
        # plot the graphs

        plt.figure(1)
        plt.subplot(211)
        for i in range(len(self.orbits)):
            plt.plot(range(len(self.orbits[i])), self.orbits[i], label=("choice " + str(i)))
        #plt.rc('lines', linewidth=2.0)
        plt.ylabel('number of agents')
        #plt.xlabel('time step')
        plt.legend()
        plt.title('number of agents vs time steps')

        plt.subplot(212)
        for i in range(len(self.exp_orbits)):
            plt.plot(range(len(self.exp_orbits[i])), self.exp_orbits[i], label=("choice " + str(i)))
        #plt.rc('lines', linewidth=2.0)
        plt.ylabel('experience of agents')
        plt.xlabel('time step')
        plt.legend()
        plt.title('experience of agents vs time steps')
        plt.show()


def main():
    steps = 100
    d = Agent_Based_Decision_Model(number_of_agents= 1000, k = 0.5, alpha = 2,
                                   utility_of_choices= [1, 5], initial_experiences= [10, 5],
                                   discount_rate=0.01, noise_standard_deviation=5000)

    for i in range(steps):
        d.step()

    d.plot()


if __name__ == "__main__":
    main()
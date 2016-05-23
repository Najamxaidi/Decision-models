import numpy as np
import matplotlib.pyplot as plt

"""
The code below concerns the the decision taken be a group of agents
between two choices
"""

class decision:
    def __init__(self, numberOfAgents, k, alpha, u_a, u_b, discount):

        self.numberOfAgents = numberOfAgents
        # for utility
        self.options = 2
        self.utility_of_a = u_a
        self.utility_of_b = u_b
        self.k = k
        self.alpha = alpha
        self.previous_exp = [0, 0]
        self.new_exp = [0,0]
        self.discount = discount
        # initially every agent has an equal probability to choose an option
        self.options_Probability = [0.5, 0.5]

        # dictionary of agent with their choices
        self.agent_choices = []
        self.orbits = [[],[]]

    # utility of a given choice
    def returnUtility(self, choice):
        if choice == 0:
            return self.utility_of_a
        else:
            return self.utility_of_b

    # convert utility to experience
    def utilityToExperience(self, utility):
        return utility

    def updateExperience(self):
        e = 1

    def step(self):
        self.agent_choices.clear()
    # each agent choose an option with a probability p
        for i in range(self.numberOfAgents):
            self.agent_choices.append(np.random.choice(range(self.options), p=self.options_Probability))

        # update experience based upon utility
        # calculate experince from current choices
        count_a = 0
        count_b = 0
        temp_a = 0
        temp_b = 0
        for i in range(self.numberOfAgents):
            if self.agent_choices[i] == 0:
                temp_a += self.returnUtility(0)
                count_a += 1
            else :
                temp_b += self.returnUtility(1)
                count_b += 1

        self.orbits[0].append(count_a)
        self.orbits[1].append(count_b)

        #new experience becomes old
        self.previous_exp[0] += self.new_exp[0]
        self.previous_exp[1] += self.new_exp[1]
        # experience for the current step is calculated
        self.new_exp[0] = temp_a + self.discount*self.previous_exp[0]
        self.new_exp[1] = temp_b + self.discount * self.previous_exp[1]

        #self.orbits[0].append(self.new_exp[0])
        #self.orbits[1].append(self.new_exp[1])

        base = ((pow((self.k + self.new_exp[0]), self.alpha)) + (pow((self.k + self.new_exp[1]), self.alpha)))
        # update probability
        self.options_Probability[0] = (pow((self.k + self.new_exp[0]), self.alpha))/ base
        self.options_Probability[1] = (pow((self.k + self.new_exp[1]), self.alpha)) / base

    def plot(self):
        # plot the graphs

        for i in range(len(self.orbits)):
            plt.plot(range(len(self.orbits[i])), self.orbits[i], label=i)
        plt.rc('lines', linewidth=2.0)
        plt.legend()
        plt.show()


def main():
    steps = 100
    d = decision(numberOfAgents = 20 , k =1, alpha = 2, u_a = 0.5, u_b = 1, discount=0.001)

    for i in range(steps):
        d.step()
        #print(d.orbits)
    d.plot()


if __name__ == "__main__":
    main()
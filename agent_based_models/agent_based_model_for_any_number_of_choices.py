import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from numpy import trapz

class Agent_Based_Decision_Model:

    """
    The code below concerns the the decision taken be a group of agents
    between multiple choices. The code follows but not completely replicates the
    equation given in the paper Dussutour (2009). An agent based model has been developed where
    multiple agents choose from the number of choices avaliable
    """

    def __init__(self, number_of_agents, k, alpha, utility_of_choices,
                 initial_experiences, discount_rate, noise_standard_deviation, use_fun_for_utilities,
                 frequencies, phase):

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
        self.previous_exp = np.array(initial_experiences)  ##np.zeros((self.options,), dtype=np.float)
        self.new_exp = np.array(initial_experiences)
        self.discount = 1 - discount_rate
        self.noise_standard_deviation = noise_standard_deviation
        self.use_fun_for_utilities=use_fun_for_utilities
        self.frequencies=frequencies
        self.phase = phase

        # initially the probablity has been set to zero
        # it will be calculated later based upon the initial experience
        self.options_Probability = np.zeros((self.options,), dtype=np.float)

        # required for plotting
        self.orbits = []
        self.exp_orbits = []
        self.orbits_utility = []
        self.aggregate_utility = []

        for i in range(self.options):
            self.orbits.append([])
            self.exp_orbits.append([])
            self.orbits_utility.append([])
            self.aggregate_utility.append([])

    def step(self, noise_flag):

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
        if noise_flag == True:
            noise = np.array(np.random.normal(0, self.noise_standard_deviation, self.options))
        else:
            noise = 0

        # previous experience is discounted
        # new experience is calculated
        # previous experience is updated
        # check equation 3.2

        self.previous_exp = self.previous_exp * self.discount  # row * perv exp
        self.new_exp = (temp_utility) + self.previous_exp + noise  ##-----
        self.previous_exp = self.new_exp

        # for plotting
        for i in range(self.options):
            self.orbits[i].append(count_of_choices[i])
            if self.new_exp[i] < 0:
                self.new_exp[i] = 0
            self.exp_orbits[i].append(self.new_exp[i])
            self.orbits_utility[i].append(temp_utility[i])
            self.aggregate_utility[i].append(self.utility_of_choices[i])

    def rotation_of_utilitities(self):
        self.utility_of_choices = np.roll(self.utility_of_choices,1)

    def plot(self):
        #print the area first
        # for i in range(self.options):
        #     print("option " + str(i) + " has area under the curve as:")
        #     print("using composite trapezoidal rule " + '{:18.5f}'.format(trapz(self.orbits[i], range(0, len(self.orbits[i])))))
        #     print("using composite Simpson's rule " + '{:18.5f}'.format(simps(soln[:, i], range(0, len(soln)))))

        # plot the graphs

        plt.figure(1)
        plt.subplot(311)
        for i in range(len(self.orbits)):
            plt.plot(range(len(self.orbits[i])), self.orbits[i], label=("choice " + str(i)))
        plt.ylim(-10, self.numberOfAgents+10)
        plt.ylabel('agents')
        plt.title('1st: number of agents vs time steps -- 2nd: experience of agents vs time steps \n'
                  '3rd: utility of agents vs time steps')

        plt.subplot(312)
        for i in range(len(self.exp_orbits)):
            plt.plot(range(len(self.exp_orbits[i])), self.exp_orbits[i], label=("choice " + str(i)))
        plt.ylabel('experience')
        plt.xlabel('time step')

        # plt.subplot(413)
        # for i in range(len(self.orbits_utility)):
        #     plt.plot(range(len(self.orbits_utility[i])), self.orbits_utility[i], label=("choice " + str(i)))
        # plt.ylabel('utility gained')
        # plt.xlabel('time step')

        t = 0
        q = 0
        plt.subplot(313)
        for i in range(len(self.aggregate_utility)):
            if t < 2:
                plt.plot(range(len(self.aggregate_utility[i])), self.aggregate_utility[i], label=("A" + str(t+1)))
                t +=1
            else:
                plt.plot(range(len(self.aggregate_utility[i])), self.aggregate_utility[i], label=("B" + str(q+1)))
                q += 1

        plt.ylabel('utility')
        plt.xlabel('time step')
        plt.legend()  #bbox_to_anchor=(1.1,0.5)
        plt.show()

    def run(self, steps, rotation_step, flag, noise_flag, plotting):
        j = 0
        for i in range(steps):
            j += 1
            if j == rotation_step and flag == True:
                self.rotation_of_utilitities()
                j = 0

            if self.use_fun_for_utilities:
                a = sine_fun(frequency=self.frequencies[0], t=i, fs=20000, phase=self.phase[0])
                b = 1 - a

                a1 = sine_fun(frequency=self.frequencies[1], t=i, fs=20000, phase=self.phase[1])
                a2 = 1 - a1

                b1 = sine_fun(frequency=self.frequencies[2], t=i, fs=20000, phase=self.phase[2])
                b2 = 1 - b1

                self.utility_of_choices = np.array([a * a1, a * a2, b * b1, b * b2])

            self.step(noise_flag)

        if plotting == True:
            self.plot()

    def return_average_utility(self,steps,rotation_step):
        self.run(steps,rotation_step, False, True, plotting=False)
        #print(self.orbits_utility)
        return np.average(np.average(self.orbits_utility,0))


def sine_fun(frequency, t, fs, phase):
    y = (np.sin(2 * np.pi * frequency * t/ fs + phase) + 1) / 2
    return y

def main():
    steps = 1000
    rotation_step = 200
    flag = False
    noise_flag = False
    d = Agent_Based_Decision_Model(number_of_agents= 100,
                                   k = 0.5, alpha = 2,
                                   utility_of_choices= [0,0,0,0],
                                   initial_experiences=[1, 1, 1, 1],
                                   discount_rate=0.1,
                                   noise_standard_deviation=10,
                                   use_fun_for_utilities=True,
                                   frequencies=[100,500,100],
                                   phase=[0,np.pi/3,np.pi/4]
                                   )

    d.run(steps,rotation_step, flag, noise_flag, plotting=True)
    # sum = 0
    # for i in range(100):
    #     sum += d.return_average_utility(steps,rotation_step)
    # print(sum / 100)



if __name__ == "__main__":
    main()
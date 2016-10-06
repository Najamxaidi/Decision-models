import numpy as np
import matplotlib.pyplot as plt

"""
This model creates hierarchy of three agent based models. The top node is called node AB. At this node all the agents
get the combined experience of choices A1 - A2 and B1 - B2. Depending upon the experiences agents decide upon the node.
Once they are in the node the decide upon the choice again depending upon the experience of the choices. They experience
direct utility at this point. This utility is used to update experience both at the current node and also at the node at
AB.

Utility changes at a different frequency in node A and in node B. Noise is used to help agents make better decisions in
the dynamic environment. There is a an optimal amount of noise needed. The optimal amount of noise maximises the total
utility gained by the agents. If there is too much noise or two little noise then that utility is not optimised.
"""


class Hierarchy:
    def __init__(self, number_of_agents, k, alpha, utility_of_choices,
                 initial_experiences, discount_rate, noise_standard_deviation, utility_flag, use_fun_for_utilities,
                 frequencies, phase):
        """

        :param number_of_agents: an integer representing the total number of agents
        :param k: threshold beyond which experience is effective
        :param alpha: senstivity of the process of choice
        :param utility_of_choices: list of utility of choices
        :param initial_experiences: list of intial experiences
        :param discount_rate: list of discount rate for each node
        :param noise_standard_deviation: list of sd for each node
        :param utility_flag: if false than swap utilities in cluster otherwise shift utilities
        :param use_fun_for_utilities: If true then use sine function for utilities rather than utilities given in utility_of_choices
        :param frequencies: frequencies of the two sine waves
        """

        self.numberOfAgents = number_of_agents
        self.k = k
        self.alpha = alpha
        self.utillity_flag = utility_flag
        self.use_fun_for_utilities = use_fun_for_utilities
        self.frequencies = frequencies
        self.phase = phase

        # if I am not using function for utilities than I can create a step function and determine how I want to rotate them
        if not use_fun_for_utilities:
            if utility_flag: # if this is true then the four utilities are shifted
                self.utility_of_choices_for_node_two = np.array(utility_of_choices)
                self.utility_of_choices_for_node_three = np.array(utility_of_choices)
            else: # else utilities are swapped
                self.utility_of_choices_for_node_two = np.array([utility_of_choices[0],utility_of_choices[1]])
                self.utility_of_choices_for_node_three = np.array([utility_of_choices[2],utility_of_choices[3]])
        else:
            a = sine_fun(frequency=self.frequencies[0], t=0, fs=8000, phase=self.phase[0])
            b = sine_fun(frequency=self.frequencies[1], t=0, fs=8000, phase=self.phase[1])

            a1 = sine_fun(frequency=self.frequencies[2], t=0, fs=8000, phase=self.phase[2])
            a2 = sine_fun(frequency=self.frequencies[3], t=0, fs=8000, phase=self.phase[3])

            b1 = sine_fun(frequency=self.frequencies[4], t=0, fs=8000, phase=self.phase[4])
            b2 = sine_fun(frequency=self.frequencies[5], t=0, fs=8000, phase=self.phase[5])

            self.utility_of_choices_for_node_one = np.array([a,b])
            self.utility_of_choices_for_node_two = np.array([a*a1, a*a2])
            self.utility_of_choices_for_node_three = np.array([b*b1, b*b2])


        # experience of each choice at node-1 is the combined experience of the two choices
        self.previous_exp_for_node_one = np.array([initial_experiences[0] + initial_experiences[1],
                                                  initial_experiences[2] + initial_experiences[3]])
        self.new_exp_for_node_one = np.array([initial_experiences[0] + initial_experiences[1],
                                                initial_experiences[2] + initial_experiences[3]])

        self.previous_exp_for_node_two = np.array([initial_experiences[0], initial_experiences[1]])
        self.new_exp_for_node_two = np.array([initial_experiences[0], initial_experiences[1]])

        self.previous_exp_for_node_three = np.array([initial_experiences[2], initial_experiences[3]])
        self.new_exp_for_node_three = np.array([initial_experiences[2], initial_experiences[3]])

        self.discount_for_node_one = 1.0 - discount_rate[0]
        self.discount_for_node_two = 1.0 - discount_rate[1]
        self.discount_for_node_three = 1.0 - discount_rate[2]

        self.noise_standard_deviation_for_node_one = noise_standard_deviation[0]
        self.noise_standard_deviation_for_node_two = noise_standard_deviation[1]
        self.noise_standard_deviation_for_node_three = noise_standard_deviation[2]

        #####-------------These parameters are internal----------------######

        self.options = 2 # need to change in order to make it generic
        self.options_Probability_for_node_one = np.zeros((self.options,), dtype=np.float)
        self.options_Probability_for_node_two = np.zeros((self.options,), dtype=np.float)
        self.options_Probability_for_node_three = np.zeros((self.options,), dtype=np.float)

        # required for plotting
        self.number_of_agents_for_one = [[],[]]
        self.number_of_agents_for_two = [[],[]]
        self.number_of_agents_for_three = [[],[]]

        # keep track of how the experience changes
        self.exp_for_one = [[],[]]
        self.exp_for_two = [[],[]]
        self.exp_for_three = [[],[]]

        # keep track of the combined utility all agents get
        self.utility_for_node_one = [[],[]]
        self.utility_for_node_two = [[],[]]
        self.utility_for_node_three = [[],[]]

        # keep tracks of the utility as it changes
        self.aggregated_utility_node1 = [[],[]]
        self.aggregated_utility_node2 = [[],[]]
        self.aggregated_utility_node3 = [[],[]]

    def step(self, noise_flag):

        ####______________HOUSE-KEEPING__________####
        # generate a noise array
        if noise_flag == True:
            noise_for_node_one = np.array(np.random.normal(0, self.noise_standard_deviation_for_node_one, self.options))
            noise_for_node_two = np.array(np.random.normal(0, self.noise_standard_deviation_for_node_two, self.options))
            noise_for_node_three = np.array(np.random.normal(0, self.noise_standard_deviation_for_node_three, self.options))
        else:
            noise_for_node_one = 0
            noise_for_node_two = 0
            noise_for_node_three = 0

        ####________________NODE-1________________####

        # Calculate the probablity of choosing a cluster based upon the initial experience
        self.options_Probability_for_node_one = (np.power((self.new_exp_for_node_one + self.k), self.alpha)) / \
                                   np.sum(np.power((self.new_exp_for_node_one + self.k), self.alpha))

        # Agent make choice between two nodes
        agent_choices_for_node_one = np.random.choice(range(self.options), self.numberOfAgents,
                                                      p=self.options_Probability_for_node_one)

        # counts how many agents have taken such a decision
        count_of_choices_for_node_one = np.zeros((self.options,), dtype=np.int)
        for i in range(self.options):
            count_of_choices_for_node_one[i] = np.size(np.where(agent_choices_for_node_one == i))

        # no utility is experienced over here. Experience will be updated later

        ####________________NODE-2________________####

        # Calculate the probablity of choosing an option based upon the initial experience
        self.options_Probability_for_node_two = (np.power((self.new_exp_for_node_two + self.k), self.alpha)) / \
                                                    np.sum(np.power((self.new_exp_for_node_two + self.k), self.alpha))

        # Agent make choice between two nodes
        agent_choices_for_node_two = np.random.choice(range(self.options), count_of_choices_for_node_one[0],
                                                          p=self.options_Probability_for_node_two)

        # counts how many agents have taken such a decision
        count_of_choices_for_node_two = np.zeros((self.options,), dtype=np.int)
        for i in range(self.options):
            count_of_choices_for_node_two[i] = np.size(np.where(agent_choices_for_node_two == i))

        # calculate the total utility of the options
        if self.utillity_flag:
            temp_utility_for_node_two = count_of_choices_for_node_two * [self.utility_of_choices_for_node_two[0],
                                                                     self.utility_of_choices_for_node_two[1]]
        else:
            temp_utility_for_node_two = count_of_choices_for_node_two * self.utility_of_choices_for_node_two

        self.previous_exp_for_node_two = self.previous_exp_for_node_two * self.discount_for_node_two
        self.new_exp_for_node_two = temp_utility_for_node_two + self.previous_exp_for_node_two + noise_for_node_two
        self.previous_exp_for_node_two = self.new_exp_for_node_two

        ####________________NODE-3________________####

        # Calculate the probablity of choosing an option based upon the initial experience
        self.options_Probability_for_node_three = (np.power((self.new_exp_for_node_three + self.k), self.alpha)) / \
                                                np.sum(np.power((self.new_exp_for_node_three + self.k), self.alpha))

        # Agent make choice between two nodes
        agent_choices_for_node_three = np.random.choice(range(self.options), count_of_choices_for_node_one[1],
                                                      p=self.options_Probability_for_node_three)

        # counts how many agents have taken such a decision
        count_of_choices_for_node_three = np.zeros((self.options,), dtype=np.int)
        for i in range(self.options):
            count_of_choices_for_node_three[i] = np.size(np.where(agent_choices_for_node_three == i))

        # calculate the total utility of the options
        if self.utillity_flag:
            temp_utility_for_node_three = count_of_choices_for_node_three * [self.utility_of_choices_for_node_three[2],
                                                                     self.utility_of_choices_for_node_three[3]]
        else:
            temp_utility_for_node_three = count_of_choices_for_node_three * self.utility_of_choices_for_node_three

        self.previous_exp_for_node_three = self.previous_exp_for_node_three * self.discount_for_node_three
        self.new_exp_for_node_three = temp_utility_for_node_three + self.previous_exp_for_node_three \
                                      + noise_for_node_three
        self.previous_exp_for_node_three = self.new_exp_for_node_three

        ####________________UPDATE VALUES FOR NODE-1________________####

        temp_utility_for_node_one = [np.sum(temp_utility_for_node_two),np.sum(temp_utility_for_node_three)]
        self.previous_exp_for_node_one = self.previous_exp_for_node_one * self.discount_for_node_one
        self.new_exp_for_node_one = temp_utility_for_node_one + self.previous_exp_for_node_one + noise_for_node_one
        self.previous_exp_for_node_one = self.new_exp_for_node_one

        # self.new_exp_for_node_one = [np.sum(self.new_exp_for_node_two), np.sum(self.new_exp_for_node_three)] \
        #                             + self.previous_exp_for_node_one + noise_for_node_one


        ####_______________PLOTTING-STUFF_______________####

        for i in range(self.options):
            self.number_of_agents_for_one[i].append(count_of_choices_for_node_one[i])
            self.utility_for_node_one[i].append(temp_utility_for_node_one)
            if self.use_fun_for_utilities:
                self.aggregated_utility_node1[i].append(self.utility_of_choices_for_node_one[i])
            if self.new_exp_for_node_one[i] < 0:
                self.exp_for_one[i].append(0)
            else:
                self.exp_for_one[i].append(self.new_exp_for_node_one[i])

            self.number_of_agents_for_two[i].append(count_of_choices_for_node_two[i])
            self.aggregated_utility_node2[i].append(self.utility_of_choices_for_node_two[i])
            self.utility_for_node_two[i].append(temp_utility_for_node_two)
            if self.new_exp_for_node_two[i] < 0:
                self.exp_for_two[i].append(0)
            else:
                self.exp_for_two[i].append(self.new_exp_for_node_two[i])

            self.number_of_agents_for_three[i].append(count_of_choices_for_node_three[i])
            self.aggregated_utility_node3[i].append(self.utility_of_choices_for_node_three[i])
            self.utility_for_node_three[i].append(temp_utility_for_node_three)
            if self.new_exp_for_node_three[i] < 0:
                self.exp_for_three[i].append(0)
            else:
                self.exp_for_three[i].append(self.new_exp_for_node_three[i])

    def run(self, steps, rotation_step, rotation_flag, noise_flag, plotting):
        j = 0
        k = 0
        l = 0
        for i in range(steps):
            j += 1
            k += 1
            l += 1

            if j == rotation_step[0] and rotation_flag == True:
                self.rotation_of_utilitities_for_node_1()
                j = 0

            if k == rotation_step[1] and rotation_flag == True:
                self.rotation_of_utilitities_for_node_2()
                k = 0

            if l == rotation_step[2] and rotation_flag == True:
                self.rotation_of_utilitities_for_node_3()
                l = 0

            ##calculate new utility
            if self.use_fun_for_utilities:
                a = sine_fun(frequency=self.frequencies[0], t=i, fs=8000, phase=self.phase[0])
                b = sine_fun(frequency=self.frequencies[1], t=i, fs=8000, phase=self.phase[1])

                a1 = sine_fun(frequency=self.frequencies[2], t=i, fs=8000, phase=self.phase[2])
                a2 = sine_fun(frequency=self.frequencies[3], t=i, fs=8000, phase=self.phase[3])

                b1 = sine_fun(frequency=self.frequencies[4], t=i, fs=8000, phase=self.phase[4])
                b2 = sine_fun(frequency=self.frequencies[5], t=i, fs=8000, phase=self.phase[5])

                self.utility_of_choices_for_node_one = np.array([a, b])
                self.utility_of_choices_for_node_two = np.array([a * a1, a * a2])
                self.utility_of_choices_for_node_three = np.array([b * b1, b * b2])

            self.step(noise_flag)

        if plotting:
            self.plot()

    def rotation_of_utilitities_for_node_1(self):
        self.utility_of_choices_for_node_two, self.utility_of_choices_for_node_three \
            = self.utility_of_choices_for_node_three, self.utility_of_choices_for_node_two

    def rotation_of_utilitities_for_node_2(self):
        self.utility_of_choices_for_node_two = np.roll(self.utility_of_choices_for_node_two,1)

    def rotation_of_utilitities_for_node_3(self):
        self.utility_of_choices_for_node_three = np.roll(self.utility_of_choices_for_node_three, 1)

    def plot(self):

        plt.figure(1)
        plt.subplot(311)
        for i in range(len(self.number_of_agents_for_one)):
            plt.plot(range(len(self.number_of_agents_for_one[i])), self.number_of_agents_for_one[i], label=(chr(97 +i)))
        plt.ylim(-10, self.numberOfAgents+10)
        plt.ylabel('number of agents')
        plt.title('number of agents/experience/utility of choices vs time steps \n NODE_AB')

        plt.subplot(312)
        for i in range(len(self.exp_for_one)):
            plt.plot(range(len(self.exp_for_one[i])), self.exp_for_one[i], label=("Cluster " + str(i)))
        plt.ylabel('experience of agents')
        #plt.xlabel('time step')

        plt.subplot(313)
        for i in range(len(self.aggregated_utility_node1)):
            plt.plot(range(len(self.aggregated_utility_node1[i])), self.aggregated_utility_node1[i],
                     label=(chr(65 +i)))
        plt.ylabel('utility of choices')
        plt.xlabel('time step')
        plt.legend()

        plt.figure(2)
        plt.subplot(311)
        for i in range(len(self.number_of_agents_for_two)):
            plt.plot(range(len(self.number_of_agents_for_two[i])), self.number_of_agents_for_two[i],
                     label=("choice " + str(i)))
        plt.ylim(-10, self.numberOfAgents+10)
        plt.ylabel('number of agents')
        plt.title('number of agents/experience/utility of choices vs time steps \n NODE_A')

        plt.subplot(312)
        for i in range(len(self.exp_for_two)):
            plt.plot(range(len(self.exp_for_two[i])), self.exp_for_two[i], label=("choice " + str(i)))
        plt.ylabel('experience of agents')
        #plt.xlabel('time step')

        plt.subplot(313)
        for i in range(len(self.aggregated_utility_node2)):
            plt.plot(range(len(self.aggregated_utility_node2[i])), self.aggregated_utility_node2[i], label=("A"+str(i+1)))
        plt.ylabel('utility of choices')
        plt.xlabel('time step')
        plt.legend()

        plt.figure(3)
        plt.subplot(311)
        for i in range(len(self.number_of_agents_for_three)):
            plt.plot(range(len(self.number_of_agents_for_three[i])), self.number_of_agents_for_three[i],
                     label=("choice " + str(i+2)))
        plt.ylim(-10, self.numberOfAgents+10)
        plt.ylabel('number of agents')
        plt.title('number of agents/experience/utility of choices vs time steps \n NODE_B')

        plt.subplot(312)
        for i in range(len(self.exp_for_three)):
            plt.plot(range(len(self.exp_for_three[i])), self.exp_for_three[i], label=("choice " + str(i+2)))
        plt.ylabel('experience of agents')
        #plt.xlabel('time step')

        plt.subplot(313)
        for i in range(len(self.aggregated_utility_node3)):
            plt.plot(range(len(self.aggregated_utility_node3[i])), self.aggregated_utility_node3[i],
                     label=("B"+str(i+1)))
        plt.ylabel('utility of choices')
        plt.xlabel('time step')
        plt.legend()
        plt.show()

    def return_average_utility_for_node1(self, steps, rotation_step,rotation_flag):
        self.run(steps, rotation_step, rotation_flag, True, plotting=False)
        return np.average(np.average(self.utility_for_node_one, 0))

    def return_average_utility_for_node2(self, steps, rotation_step,rotation_flag):
        self.run(steps, rotation_step, rotation_flag, True, plotting=False)
        return np.average(np.average(self.utility_for_node_two, 0))

    def return_average_utility_for_node3(self, steps, rotation_step,rotation_flag):
        self.run(steps, rotation_step, rotation_flag, True, plotting=False)
        return np.average(np.average(self.utility_for_node_three, 0))


def sine_fun(frequency, t, fs, phase):
    y = (np.sin(2 * np.pi * frequency * t/ fs + phase) + 1) / 2
    return y

def main():
    steps = 1000
    rotation_step = [400, 200, 200]
    rotation_flag = False
    noise_flag = True

    d = Hierarchy(number_of_agents= 100,
                  k = 1, alpha = 2,
                  utility_of_choices= [0,0,0,0],
                  initial_experiences=[1, 1, 1, 1],
                  discount_rate=[1,1,1],
                  noise_standard_deviation=[8,12,0.1],
                  utility_flag = False, # if false than swap utilities in cluster otherwise shift utilities
                  use_fun_for_utilities = True, # utility flag should be false when using it
                  frequencies=[100,100,100,600,200,500],
                  phase=[0,np.pi,np.pi/2,np.pi/3,np.pi/4,np.pi/5])

    d.run(steps,rotation_step, rotation_flag, noise_flag, plotting=True)

if __name__ == "__main__":
    main()
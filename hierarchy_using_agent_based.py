import numpy as np
import matplotlib.pyplot as plt


class Hierarchy:
    def __init__(self, number_of_agents, k, alpha, utility_of_choices,
                 initial_experiences, discount_rate, noise_standard_deviation):

        self.numberOfAgents = number_of_agents
        self.k = k
        self.alpha = alpha

        self.utility_of_choices_for_node_two = np.array(utility_of_choices)
        self.utility_of_choices_for_node_three = np.array(utility_of_choices)

        self.previous_exp_for_node_one = np.array([initial_experiences[0] + initial_experiences[1],
                                                  initial_experiences[2] + initial_experiences[3]])
        self.new_exp_for_node_one = np.array([initial_experiences[0] + initial_experiences[1],
                                                initial_experiences[2] + initial_experiences[3]])

        self.previous_exp_for_node_two = np.array([initial_experiences[0], initial_experiences[1]])
        self.new_exp_for_node_two = np.array([initial_experiences[0], initial_experiences[1]])

        self.previous_exp_for_node_three = np.array([initial_experiences[2], initial_experiences[3]])
        self.new_exp_for_node_three = np.array([initial_experiences[2], initial_experiences[3]])

        self.discount_for_node_one = discount_rate[0]
        self.discount_for_node_two = discount_rate[1]
        self.discount_for_node_three = discount_rate[2]

        self.noise_standard_deviation_for_node_one = noise_standard_deviation[0]
        self.noise_standard_deviation_for_node_two = noise_standard_deviation[1]
        self.noise_standard_deviation_for_node_three = noise_standard_deviation[2]

        #####-------------These parameters are internal----------------######

        self.options = 2
        self.options_Probability_for_node_one = np.zeros((self.options,), dtype=np.float)
        self.options_Probability_for_node_two = np.zeros((self.options,), dtype=np.float)
        self.options_Probability_for_node_three = np.zeros((self.options,), dtype=np.float)

        # required for plotting
        self.number_of_agents_for_one = [[],[]]
        self.number_of_agents_for_two = [[],[]]
        self.number_of_agents_for_three = [[],[]]

        self.exp_for_one = [[], []]
        self.exp_for_two = [[], []]
        self.exp_for_three = [[], []]

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


        ####________________NODE-2________________####

        # Calculate the probablity of choosing a cluster based upon the initial experience
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
        temp_utility_for_node_two = count_of_choices_for_node_two * [self.utility_of_choices_for_node_two[0],
                                                                     self.utility_of_choices_for_node_two[1]]

        self.previous_exp_for_node_two *= self.discount_for_node_two
        self.new_exp_for_node_two = (temp_utility_for_node_two) + self.previous_exp_for_node_two + noise_for_node_two

        ####________________NODE-3________________####

        # Calculate the probablity of choosing a cluster based upon the initial experience
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
        temp_utility_for_node_three = count_of_choices_for_node_three * [self.utility_of_choices_for_node_three[2],
                                                                     self.utility_of_choices_for_node_three[3]]

        self.previous_exp_for_node_three *= self.discount_for_node_three
        self.new_exp_for_node_three = (temp_utility_for_node_three) + self.previous_exp_for_node_three \
                                      + noise_for_node_three


        ####________________UPDATE VALUES FOR NODE-1________________####

        self.previous_exp_for_node_one *= self.discount_for_node_one
        self.new_exp_for_node_one = [np.sum(self.new_exp_for_node_two),np.sum(self.new_exp_for_node_three)] \
                                    + self.previous_exp_for_node_one + noise_for_node_one


        ####_______________PLOTTING-STUFF_______________####

        for i in range(self.options):
            self.number_of_agents_for_one[i].append(count_of_choices_for_node_one[i])
            self.exp_for_one[i].append(self.new_exp_for_node_one[i])

            self.number_of_agents_for_two[i].append(count_of_choices_for_node_two[i])
            self.exp_for_two[i].append(self.new_exp_for_node_two[i])

            self.number_of_agents_for_three[i].append(count_of_choices_for_node_three[i])
            self.exp_for_three[i].append(self.new_exp_for_node_three[i])

    def run(self, steps, rotation_step, rotation_flag, noise_flag, plotting):
        j = 0
        k = 0
        for i in range(steps):
            j += 1
            k += 1
            if j == rotation_step[0] and rotation_flag == True:
                self.rotation_of_utilitities_for_node_2()
                j = 0

            if k == rotation_step[1] and rotation_flag == True:
                self.rotation_of_utilitities_for_node_3()
                k = 0

            self.step(noise_flag)

        if plotting == True:
            self.plot()

    def rotation_of_utilitities_for_node_2(self):
        self.utility_of_choices_for_node_two = np.roll(self.utility_of_choices_for_node_two,1)

    def rotation_of_utilitities_for_node_3(self):
        self.utility_of_choices_for_node_three = np.roll(self.utility_of_choices_for_node_three, 1)

    def plot(self):

        plt.figure(1)
        plt.subplot(211)
        for i in range(len(self.number_of_agents_for_one)):
            plt.plot(range(len(self.number_of_agents_for_one[i])), self.number_of_agents_for_one[i], label=("Cluster " + str(i)))
        plt.ylim(-10, self.numberOfAgents+10)
        plt.ylabel('number of agents')
        plt.title('1st: number of agents vs time steps -- 2nd: experience of agents vs time steps \n NODE_1')

        plt.subplot(212)
        for i in range(len(self.exp_for_one)):
            plt.plot(range(len(self.exp_for_one[i])), self.exp_for_one[i], label=("Cluster " + str(i)))
        plt.ylabel('experience of agents')
        plt.xlabel('time step')
        plt.legend()
        #plt.show()

        plt.figure(2)
        plt.subplot(211)
        for i in range(len(self.number_of_agents_for_two)):
            plt.plot(range(len(self.number_of_agents_for_two[i])), self.number_of_agents_for_two[i],
                     label=("choice " + str(i)))
        plt.ylim(-10, self.numberOfAgents+10)
        plt.ylabel('number of agents')
        plt.title('1st: number of agents vs time steps -- 2nd: experience of agents vs time steps \n NODE_2')

        plt.subplot(212)
        for i in range(len(self.exp_for_two)):
            plt.plot(range(len(self.exp_for_two[i])), self.exp_for_two[i], label=("choice " + str(i)))
        plt.ylabel('experience of agents')
        plt.xlabel('time step')
        plt.legend()

        plt.figure(3)
        plt.subplot(211)
        for i in range(len(self.number_of_agents_for_three)):
            plt.plot(range(len(self.number_of_agents_for_three[i])), self.number_of_agents_for_three[i],
                     label=("choice " + str(i+2)))
        plt.ylim(-10, self.numberOfAgents+10)
        plt.ylabel('number of agents')
        plt.title('1st: number of agents vs time steps -- 2nd: experience of agents vs time steps \n NODE_3')

        plt.subplot(212)
        for i in range(len(self.exp_for_three)):
            plt.plot(range(len(self.exp_for_three[i])), self.exp_for_three[i], label=("choice " + str(i+2)))
        plt.ylabel('experience of agents')
        plt.xlabel('time step')
        plt.legend()
        plt.show()


def main():
    steps = 1000
    rotation_step = [200, 100]
    rotation_flag = True
    noise_flag = True
    d = Hierarchy(number_of_agents= 100,
                  k = 1, alpha = 2,
                  utility_of_choices= [50,20,150,70],
                  initial_experiences=[20, 150, 10, 30],
                  discount_rate=[1,1,1],
                  noise_standard_deviation=[100,100,100])

    d.run(steps,rotation_step, rotation_flag, noise_flag, plotting=True)


if __name__ == "__main__":
    main()
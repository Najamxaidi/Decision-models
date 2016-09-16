import agent_based_model_for_any_number_of_choices as agentb
import hierarchy_using_agent_based as agenth
import matplotlib.pyplot as plt
import numpy as np


class AgentBasedStatGenerator:

    def __init__(self,number_of_agents, k, alpha, utility_of_choices, initial_experiences,
                 discount_rate = 0.0, step_for_sd = 0.01, end_sd = 10):

        self.number_of_agents = number_of_agents
        self.k = k
        self.alpha = alpha
        self.utility_of_choices = np.array(utility_of_choices)
        self.experiences_of_choices = np.array(initial_experiences)
        self.discount_rate = discount_rate
        self.step_for_sd = step_for_sd
        self.end_sd = end_sd

    def generate_utility_sweep(self, steps, rotation_step):
        utility_array = []
        standard_deviation = []
        ######### Generating statistics #############
        for i in self.frange(0.1, self.end_sd, self.step_for_sd):
            agb = agentb.Agent_Based_Decision_Model(number_of_agents=self.number_of_agents,
                                                    k=self.k,
                                                    alpha=self.alpha,
                                                    utility_of_choices=self.utility_of_choices,
                                                    initial_experiences=self.experiences_of_choices,
                                                    discount_rate=self.discount_rate,
                                                    noise_standard_deviation=i
                                                    )

            utility_array.append(agb.return_average_utility(steps, rotation_step))
            standard_deviation.append(i)

        plt.plot(standard_deviation, utility_array, label="test")
        plt.xlabel('standard deviation')
        plt.ylabel('average utility')
        # plt.legend()
        plt.show()

    def generate_utility_sweep_using_hierarchy(self, steps, rotation_step,util_flag):
        utility_array_for_node_2 = []
        utility_array_for_node_3 = []
        standard_deviation = []
        ######### Generating statistics #############
        for i in self.frange(0.1, self.end_sd, self.step_for_sd):
            agb = agenth.Hierarchy(number_of_agents= self.number_of_agents,
                                    k = self.k, alpha = self.alpha,
                                    utility_of_choices = self.utility_of_choices,
                                    initial_experiences=self.experiences_of_choices,
                                    discount_rate=[self.discount_rate,self.discount_rate,self.discount_rate],
                                    noise_standard_deviation=[i,i,i],
                                    utility_flag = util_flag)

            utility_array_for_node_2.append(agb.return_average_utility_for_node2(steps, rotation_step))
            utility_array_for_node_3.append(agb.return_average_utility_for_node3(steps, rotation_step))
            standard_deviation.append(i)

        plt.figure(1)
        plt.subplot(211)
        plt.plot(standard_deviation, utility_array_for_node_2, label="test")
        plt.xlabel('standard deviation')
        plt.ylabel('average utility')

        plt.subplot(212)
        plt.plot(standard_deviation, utility_array_for_node_3, label="test")
        plt.xlabel('standard deviation')
        plt.ylabel('average utility')
        plt.show()

    def frange(self, start, stop, step):
        i = start
        while i < stop:
            yield i
            i += step

def main():

    absg = AgentBasedStatGenerator(number_of_agents=100,
                                    k=1,
                                    alpha=2,
                                    utility_of_choices=[50,50,1,99],
                                    initial_experiences=[50, 50, 100, 0],
                                    discount_rate=1,
                                    step_for_sd=10,
                                    end_sd=5000)


    steps = 1000
    rotation_step = [100,400]

    absg.generate_utility_sweep_using_hierarchy(steps,rotation_step, util_flag=False)

if __name__ == "__main__":
    main()


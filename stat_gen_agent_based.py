import agent_based_model_for_any_number_of_choices as agentb
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

    def frange(self, start, stop, step):
        i = start
        while i < stop:
            yield i
            i += step

def main():

    absg = AgentBasedStatGenerator(number_of_agents=100,
                                    k=1,
                                    alpha=2,
                                    utility_of_choices=[1.5, 3.0],
                                    initial_experiences=[0.001, 0.003],
                                    discount_rate=0.95,
                                    step_for_sd=0.5,
                                    end_sd=1000)


    steps = 1000
    rotation_step = 200

    absg.generate_utility_sweep(steps,rotation_step)

if __name__ == "__main__":
    main()


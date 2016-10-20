import matplotlib.pyplot as plt
import numpy as np

import agent_based_models.hierarchy_using_agent_based as agenth
from agent_based_models import agent_based_model_for_any_number_of_choices as agentb


class AgentBasedStatGenerator:

    def __init__(self,number_of_agents, k, alpha, utility_of_choices, initial_experiences,
                 discount_rate, step_for_sd, end_sd, utility_flag, use_fun_for_utilities):

        self.number_of_agents = number_of_agents
        self.k = k
        self.alpha = alpha
        self.utility_of_choices = np.array(utility_of_choices)
        self.experiences_of_choices = np.array(initial_experiences)
        self.discount_rate = discount_rate
        self.step_for_sd = step_for_sd
        self.end_sd = end_sd
        self.utility_flag = utility_flag
        self.use_fun_for_utilities = use_fun_for_utilities

    def generate_utility_sweep_using_flat_model(self, steps, rotation_step, feq, phase):
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
                                                    noise_standard_deviation=i,
                                                    use_fun_for_utilities=self.use_fun_for_utilities,
                                                    frequencies=feq,
                                                    phase= phase
                                                    )

            utility_array.append(agb.return_average_utility(steps, rotation_step))
            standard_deviation.append(i)

        plt.plot(standard_deviation, utility_array, label="test")
        plt.xlabel('standard deviation')
        plt.ylabel('average utility')
        plt.show()

    def generate_utility_sweep_using_hierarchy(self, steps, rotation_step,rotation_flag,frequencies,phase):
        utility_array_for_node_1 = []
        utility_array_for_node_2 = []
        utility_array_for_node_3 = []
        utility_array = []
        standard_deviation = []
        ######### Generating statistics #############
        for i in self.frange(0.1, self.end_sd, self.step_for_sd):
            agb = agenth.Hierarchy(number_of_agents= self.number_of_agents,
                                    k = self.k, alpha = self.alpha,
                                    utility_of_choices = self.utility_of_choices,
                                    initial_experiences=self.experiences_of_choices,
                                    discount_rate=self.discount_rate,
                                    noise_standard_deviation=[i,i,i],
                                    utility_flag = self.utility_flag,
                                    use_fun_for_utilities=self.use_fun_for_utilities, # utility flag should be false when using it
                                    frequencies=frequencies,
                                    phase=phase)

            utility_array_for_node_1.append(agb.return_average_utility_for_node1(steps, rotation_step, rotation_flag))
            utility_array_for_node_2.append(agb.return_average_utility_for_node2(steps, rotation_step,rotation_flag))
            utility_array_for_node_3.append(agb.return_average_utility_for_node3(steps, rotation_step,rotation_flag))
            utility_array.append(agb.return_average_utility(steps, rotation_step, rotation_flag))
            standard_deviation.append(i)

        # total_utility_average_array = np.array(utility_array_for_node_1) + np.array(utility_array_for_node_2) \
        #                               + np.array(utility_array_for_node_3)
        #
        # total_utility_average_array =  total_utility_average_array/3

        plt.figure(1)
        plt.subplot(411)
        plt.plot(standard_deviation, utility_array_for_node_1, label="test")
        plt.xlabel('standard deviation')
        plt.ylabel('average utility \n at node AB')

        plt.subplot(412)
        plt.plot(standard_deviation, utility_array_for_node_2, label="test")
        plt.xlabel('standard deviation')
        plt.ylabel('average utility \n at node A')

        plt.subplot(413)
        plt.plot(standard_deviation, utility_array_for_node_3, label="test")
        plt.xlabel('standard deviation')
        plt.ylabel('average utility  \n at node B')

        plt.subplot(414)
        plt.plot(standard_deviation, utility_array, label="test")
        plt.xlabel('standard deviation')
        plt.ylabel('total average \n utility')
        plt.show()

    def average_run_using_hierarchy(self,number_of_runs, steps, rotation_step, rotation_flag, noise_flag, sd, frequencies, phase):

        average_number_of_agents_for_one = np.array([np.zeros(steps), np.zeros(steps)])
        average_number_of_agents_for_two = np.array([np.zeros(steps), np.zeros(steps)])
        average_number_of_agents_for_three = np.array([np.zeros(steps), np.zeros(steps)])

        average_exp_for_one = np.array([np.zeros(steps), np.zeros(steps)])
        average_exp_for_two = np.array([np.zeros(steps), np.zeros(steps)])
        average_exp_for_three = np.array([np.zeros(steps), np.zeros(steps)])

        for i in range(number_of_runs):
            d = agenth.Hierarchy(number_of_agents= self.number_of_agents,
                                    k = self.k, alpha = self.alpha,
                                    utility_of_choices = self.utility_of_choices,
                                    initial_experiences=self.experiences_of_choices,
                                    discount_rate=self.discount_rate,
                                    noise_standard_deviation=sd,
                                    utility_flag = self.utility_flag,  # if false than swap utilities in cluster otherwise shift utilities
                                    use_fun_for_utilities=self.use_fun_for_utilities,  # utility flag should be false when using it
                                    frequencies=frequencies,
                                    phase = phase)

            d.run(steps, rotation_step, rotation_flag, noise_flag, plotting=False)

            average_number_of_agents_for_one[0] += np.array(d.number_of_agents_for_one[0])
            average_number_of_agents_for_one[1] += np.array(d.number_of_agents_for_one[1])

            average_number_of_agents_for_two[0] += np.array(d.number_of_agents_for_two[0])
            average_number_of_agents_for_two[1] += np.array(d.number_of_agents_for_two[1])

            average_number_of_agents_for_three[0] += np.array(d.number_of_agents_for_three[0])
            average_number_of_agents_for_three[1] += np.array(d.number_of_agents_for_three[1])

            average_exp_for_one[0] += np.array(d.exp_for_one[0])
            average_exp_for_one[1] += np.array(d.exp_for_one[1])

            average_exp_for_two[0] += np.array(d.exp_for_two[0])
            average_exp_for_two[1] += np.array(d.exp_for_two[1])

            average_exp_for_three[0] += np.array(d.exp_for_three[0])
            average_exp_for_three[1] += np.array(d.exp_for_three[1])


        average_number_of_agents_for_one = average_number_of_agents_for_one / number_of_runs
        average_number_of_agents_for_two = average_number_of_agents_for_two / number_of_runs
        average_number_of_agents_for_three = average_number_of_agents_for_three / number_of_runs

        average_exp_for_one = average_exp_for_one / number_of_runs
        average_exp_for_two = average_exp_for_two / number_of_runs
        average_exp_for_three = average_exp_for_three / number_of_runs

        plt.figure(1)
        plt.subplot(311)
        for i in range(len(average_number_of_agents_for_one)):
            plt.plot(range(len(average_number_of_agents_for_one[i])), average_number_of_agents_for_one[i],
                     label=(chr(65 + i)))
        plt.ylabel('number of agents \n at node AB')
        plt.legend()
        plt.title('number of agents vs time steps')

        plt.subplot(312)
        for i in range(len(average_number_of_agents_for_two)):
            plt.plot(range(len(average_number_of_agents_for_two[i])), average_number_of_agents_for_two[i],
                     label=("A"+str(i+1)))
        plt.legend()
        plt.ylabel('number of agents \n at node A')

        plt.subplot(313)
        for i in range(len(average_number_of_agents_for_three)):
            plt.plot(range(len(average_number_of_agents_for_three[i])), average_number_of_agents_for_three[i],
                     label=("B"+str(i+1)))
        plt.legend()
        plt.ylabel('number of agents \n at node B')

        plt.figure(2)
        plt.subplot(311)
        for i in range(len(average_exp_for_one)):
            plt.plot(range(len(average_exp_for_one[i])), average_exp_for_one[i],
                     label=(chr(65 + i)))
        plt.ylabel('experience of agents \n at node AB')
        plt.legend()
        plt.title('experience of agents vs time steps')

        plt.subplot(312)
        for i in range(len(average_exp_for_two)):
            plt.plot(range(len(average_exp_for_two[i])), average_exp_for_two[i],
                     label=("A"+str(i+1)))
        plt.legend()
        plt.ylabel('experience of agents \n at node A')

        plt.subplot(313)
        for i in range(len(average_exp_for_three)):
            plt.plot(range(len(average_exp_for_three[i])), average_exp_for_three[i],
                     label=("B"+str(i+1)))
        plt.ylabel('experience of agents \n at node B')
        plt.legend()
        plt.show()

    def frange(self, start, stop, step):
        i = start
        while i < stop:
            yield i
            i += step

def main():

    absg = AgentBasedStatGenerator( number_of_agents=100,
                                    k=0.5,
                                    alpha=2,
                                    utility_of_choices=[0,0,0,0],
                                    initial_experiences=[1, 1, 1, 1],
                                    discount_rate= [0.1,0.1,0.1],
                                    step_for_sd=0.5,
                                    end_sd=100,
                                    utility_flag=False,
                                    use_fun_for_utilities=True)

    # absg.generate_utility_sweep_using_flat_model(steps=1000,
    #                                              rotation_step=[0,0,0,0],
    #                                              feq=[200,1000,1000],
    #                                              phase=[0,0,0])

    absg.generate_utility_sweep_using_hierarchy(steps = 1000,
                                                rotation_step = [0,0,0],
                                                rotation_flag=False,
                                                frequencies=[200,1000,1000],
                                                phase=[0,0,0]
                                                )

    # absg.average_run_using_hierarchy(number_of_runs=100,
    #                                  steps=5000,
    #                                  rotation_step=[100,200,300],
    #                                  rotation_flag=False,
    #                                  noise_flag=False,
    #                                  sd=[0,0,0],
    #                                  frequencies=[200,400,400],
    #                                  phase=[0,0,0])

if __name__ == "__main__":
    main()


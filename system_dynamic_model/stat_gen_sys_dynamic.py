import matplotlib.pyplot as plt
import numpy as np
from system_dynamic_model import system_dynamics_using_sdeint as systemd
from system_dynamic_model import hierarchy_using_sys_dynamics as syshiery


class SystemDynamicsStatGenerator:

    def __init__(self,number_of_agents, k, alpha, utility_of_choices, initial_experiences, frequencies, phase,
                 discount_rate, step, end_sd, rotation_time, rotation_flag = False, utility_flag=False,
                                    use_fun_for_utilities=True):

        self.number_of_agents = number_of_agents
        self.k = k
        self.alpha = alpha
        self.utility_of_choices = np.array(utility_of_choices)
        self.experiences_of_choices = np.array(initial_experiences)
        self.discount_rate = discount_rate
        self.step = step
        self.end_sd = end_sd
        self.rotation_time = rotation_time
        self.utility_flag = utility_flag # how do I want rotation
        self.use_fun_for_utilities = use_fun_for_utilities
        self.frequencies = frequencies
        self.phase = phase
        self.rotation_flag = rotation_flag  # do I want rotation

    def generate_utility_sweep(self,time_vector = np.linspace(0, 1000, 500)):
        utility_array = []
        standard_deviation = []
        ######### Generating statistics #############
        for i in self.frange(0.0, self.end_sd, self.step):
            sysd = systemd.SystemDynamicsWithSdeint(number_of_agents=self.number_of_agents,
                                                    k=self.k,
                                                    alpha=self.alpha,
                                                    utility_of_choices=self.utility_of_choices,
                                                    initial_experiences=self.experiences_of_choices,
                                                    discount_rate=self.discount_rate,
                                                    sd=i,
                                                    rotation_time=self.rotation_time,
                                                    rotation_flag=self.rotation_flag,
                                                    use_fun_for_utilities=self.use_fun_for_utilities,
                                                    frequencies=self.frequencies,
                                                    phase=self.phase
                                                    )

            utility_array.append(sysd.return_average_utility(time_vector))
            standard_deviation.append(i)

        plt.plot(standard_deviation, utility_array, label="test")
        plt.xlabel('standard deviation')
        plt.ylabel('average utility')
        # plt.legend()
        plt.show()

    def generate_utility_sweep_using_hierarchy(self, time_vector, frequencies, phase):
        utility_array_for_node_2 = []
        utility_array_for_node_3 = []
        standard_deviation = []
        ######### Generating statistics #############
        for i in self.frange(0.1, self.end_sd, self.step):
            agb = syshiery.SystemDynamicsWithSdeint(number_of_agents= self.number_of_agents,
                                    k = self.k, alpha = self.alpha,
                                    utility_of_choices = self.utility_of_choices,
                                    initial_experiences=self.experiences_of_choices,
                                    discount_rate=self.discount_rate,
                                    noise_standard_deviation=[i,i,i],
                                    rotation_time=self.rotation_time,
                                    rotation_flag=self.rotation_flag,
                                    use_fun_for_utilities=self.use_fun_for_utilities, # utility flag should be false when using it
                                    frequencies=frequencies,
                                    phase=phase,
                                    utility_flag=self.utility_flag)

            utility_array_for_node_2.append(agb.return_average_utility_for_node2(time_vector))
            utility_array_for_node_3.append(agb.return_average_utility_for_node3(time_vector))
            standard_deviation.append(i)

        plt.figure(1)
        plt.subplot(211)
        plt.plot(standard_deviation, utility_array_for_node_2, label="test")
        plt.xlabel('standard deviation')
        plt.ylabel('average utility \n at node A')

        plt.subplot(212)
        plt.plot(standard_deviation, utility_array_for_node_3, label="test")
        plt.xlabel('standard deviation')
        plt.ylabel('average utility  \n at node B')
        plt.show()

    def frange(self, start, stop, step):
        i = start
        while i < stop:
            yield i
            i += step

def main():

    sdsg = SystemDynamicsStatGenerator(number_of_agents=100,
                                    k=100,
                                    alpha=2,
                                    utility_of_choices=[0,0,0,0],
                                    initial_experiences=[1,1,1,1],
                                    frequencies=[50,10,30],
                                    phase = [0,0,0],
                                    discount_rate=[1,1,1],
                                    step=0.5,
                                    end_sd=30,
                                    rotation_time=[0,0,0],
                                    rotation_flag = False,
                                    utility_flag=False,
                                    use_fun_for_utilities=True)

    #sdsg.generate_utility_sweep(time_vector = np.linspace(0, 1000, 500))

    sdsg.generate_utility_sweep_using_hierarchy(time_vector = np.linspace(0, 1000, 500), frequencies=[50, 10,30],
                                                phase=[0,0,0])

if __name__ == "__main__":
    main()


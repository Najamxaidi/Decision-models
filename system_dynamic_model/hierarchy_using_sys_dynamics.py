import numpy as np
import sdeint
import matplotlib.pyplot as plt
from scipy.integrate import simps
from numpy import trapz
from scipy.optimize import basinhopping


class SystemDynamicsWithSdeint:
    """
    This class replicates the model in Dussutour (2009)
    This follows an ODE model, see equation 3.2 in the model
    """

    def __init__(self, number_of_agents, k, alpha, utility_of_choices, initial_experiences,
                 discount_rate, noise_standard_deviation, rotation_time, rotation_flag, use_fun_for_utilities,
                 frequencies, phase, utility_flag):
        """

        :param number_of_agents: an integer with the total number of agents
        :param k: see equation 3.1
        :param alpha: see equation 3.1
        :param utility_of_choices: an array telling about how good each choice is
        :param initial_experiences: an array telling the initial experiences of choices
        :param discount_rate: a number between 0 and 1
        """

        self.number_of_agents = number_of_agents
        self.k = k
        self.alpha = alpha

        #utility
        if not use_fun_for_utilities:
            if utility_flag: # if this is true then the four utilities are shifted
                self.utility_of_choices_for_node_two = np.array(utility_of_choices)
                self.utility_of_choices_for_node_three = np.array(utility_of_choices)
            else: # else utilities are swapped
                self.utility_of_choices_for_node_two = np.array([utility_of_choices[0],utility_of_choices[1]])
                self.utility_of_choices_for_node_three = np.array([utility_of_choices[2],utility_of_choices[3]])
        else:
            a = sine_fun(frequency=frequencies[0], t=0, fs=8000, phase=phase[0])
            b = 1 - a

            a1 = sine_fun(frequency=frequencies[1], t=0, fs=8000, phase=phase[1])
            a2 = 1 - a1

            b1 = sine_fun(frequency=frequencies[2], t=0, fs=8000, phase=phase[2])
            b2 = 1 - b1

            self.utility_of_choices_for_node_one = np.array([a,b])
            self.utility_of_choices_for_node_two = np.array([a*a1, a*a2])
            self.utility_of_choices_for_node_three = np.array([b*b1, b*b2])

        #experience
        # experience of each choice at node-1 is the combined experience of the two choices
        self.experiences_at_node_1 = np.array([initial_experiences[0] + initial_experiences[1],
                                                 initial_experiences[2] + initial_experiences[3]])

        self.experiences_of_choices = np.array([initial_experiences[0], initial_experiences[1],
                                                initial_experiences[2], initial_experiences[3]])

        self.experiences_at_node_2 = []
        self.experiences_at_node_3 = []
        # discount rate
        self.discount_rate = discount_rate

        # noise_standard_deviation
        self.noise_standard_deviation_for_node_one = noise_standard_deviation[0]
        self.noise_standard_deviation_for_node_two = noise_standard_deviation[1]
        self.noise_standard_deviation_for_node_three = noise_standard_deviation[2]

        self.rotation_time = rotation_time
        self.rotation_flag = rotation_flag
        self.use_fun_for_utilities = use_fun_for_utilities
        self.frequencies = frequencies
        self.phase = phase
        self.utillity_flag = utility_flag

        #####-------------These parameters are internal----------------######
        self.options = 2
        self.count_1 = 0
        self.count_2 = 0
        self.count_3 = 0
        self.value_for_sine = 0

        # required for plotting
        self.propotion_of_agents_at_one = [[],[]]
        self.propotion_of_agents_at_two = [[],[]]
        self.propotion_of_agents_at_three = [[],[]]

        self.utility_gained_at_node_two = [[],[]]
        self.utility_gained_at_node_three = [[],[]]

        self.orbits_for_exp_at_node_1 = [[],[]]

        # keep tracks of the utility as it changes
        self.aggregated_utility_node1 = [[], []]
        self.aggregated_utility_node2 = [[], []]
        self.aggregated_utility_node3 = [[], []]

    def rotation_of_utilitities_for_node_1(self):
        self.utility_of_choices_for_node_two, self.utility_of_choices_for_node_three \
            = self.utility_of_choices_for_node_three, self.utility_of_choices_for_node_two

    def rotation_of_utilitities_for_node_2(self):
        self.utility_of_choices_for_node_two = np.roll(self.utility_of_choices_for_node_two,1)

    def rotation_of_utilitities_for_node_3(self):
        self.utility_of_choices_for_node_three = np.roll(self.utility_of_choices_for_node_three, 1)

    def rate_of_experience(self, experience, t):

        # keep track of rotation
        self.count_1 += 1
        self.count_2 += 1
        self.count_3 += 1
        if self.count_1 == self.rotation_time[0] and self.rotation_flag:
            self.rotation_of_utilitities_for_node_1()
            self.count_1 = 0
        if self.count_2 == self.rotation_time[1] and self.rotation_flag:
            self.rotation_of_utilitities_for_node_2()
            self.count_2 = 0
        if self.count_3 == self.rotation_time[2] and self.rotation_flag:
            self.rotation_of_utilitities_for_node_3()
            self.count_3 = 0

        # calculate experience of agents at the first node
        self.experiences_at_node_1[0] = experience[0] + experience[1]
        self.experiences_at_node_1[1] = experience[2] + experience[3]

        ####Plotting stuff
        for i in range(self.options):
            self.orbits_for_exp_at_node_1[i].append(self.experiences_at_node_1[i])

        # update utility accordingly
        if self.use_fun_for_utilities:
            a = sine_fun(frequency=self.frequencies[0], t=self.value_for_sine, fs=8000, phase=self.phase[0])
            b = 1 - a

            a1 = sine_fun(frequency=self.frequencies[1], t=self.value_for_sine, fs=8000, phase=self.phase[1])
            a2 = 1 - a1

            b1 = sine_fun(frequency=self.frequencies[2], t=self.value_for_sine, fs=8000, phase=self.phase[2])
            b2 = 1 - b1

            self.utility_of_choices_for_node_one = np.array([a, b])
            self.utility_of_choices_for_node_two = np.array([a * a1, a * a2])
            self.utility_of_choices_for_node_three = np.array([b * b1, b * b2])

        self.value_for_sine += 1

        ################################ NODE-1
        # calculate the probability based upon the initial experiences at node 1
        options_probability_for_node_1 = (np.power((self.experiences_at_node_1 + self.k), self.alpha)) / \
                              np.sum(np.power((self.experiences_at_node_1 + self.k), self.alpha))

        # Equation 3.1 ends
        pi_flux_for_node_1 = options_probability_for_node_1 * self.number_of_agents

        ################################ NODE-2
        exp_1 = np.array([experience[0], experience[1]])
        # calculate the probability based upon the initial experiences at node 2
        options_probability_for_node_2 = (np.power((exp_1 + self.k), self.alpha)) / \
                                         np.sum(np.power((exp_1 + self.k), self.alpha))

        # Equation 3.1 ends
        pi_qi_flux_for_node_2 = options_probability_for_node_2 * self.utility_of_choices_for_node_two \
                                * pi_flux_for_node_1[0]

        ################################ NODE-3
        exp_2 = np.array([experience[2], experience[3]])
        # calculate the probability based upon the initial experiences at node 2
        options_probability_for_node_3 = (np.power((exp_2 + self.k), self.alpha)) / \
                                         np.sum(np.power((exp_2 + self.k), self.alpha))

        # print(options_probability_for_node_1)
        # print(options_probability_for_node_2)
        # print(options_probability_for_node_3)
        # Equation 3.1 ends
        pi_qi_flux_for_node_3 = options_probability_for_node_3 * self.utility_of_choices_for_node_three \
                                * pi_flux_for_node_1[1]

        ################################ PLOTTING

        for i in range(self.options):
            self.propotion_of_agents_at_one[i].append(pi_flux_for_node_1[i])
            self.propotion_of_agents_at_two[i].append(pi_qi_flux_for_node_2[i])
            self.propotion_of_agents_at_three[i].append(pi_qi_flux_for_node_3[i])

            self.utility_gained_at_node_two[i].append(pi_qi_flux_for_node_2)
            self.utility_gained_at_node_three[i].append(pi_qi_flux_for_node_3)

            if self.use_fun_for_utilities:
                self.aggregated_utility_node1[i].append(self.utility_of_choices_for_node_one[i])
            self.aggregated_utility_node2[i].append(self.utility_of_choices_for_node_two[i])
            self.aggregated_utility_node3[i].append(self.utility_of_choices_for_node_three[i])

        ############################ UPDATE EXPERIENCES
        row_ci_1 = self.discount_rate[0] * self.experiences_at_node_1
        row_ci_2 = self.discount_rate[1] * exp_1
        row_ci_3 = self.discount_rate[2] * exp_2

        experience[0], experience[1] = pi_qi_flux_for_node_2 - row_ci_2
        experience[2], experience[3] = pi_qi_flux_for_node_3 - row_ci_3

        return experience

    def noise(self, W, t):
        noise =  np.array([self.noise_standard_deviation_for_node_two,
                    self.noise_standard_deviation_for_node_two,
                    self.noise_standard_deviation_for_node_three,
                    self.noise_standard_deviation_for_node_three])
        B = np.diag(noise)
        return B

    def solve(self, time_vector=np.linspace(0, 10, 10000)):

        t = time_vector
        soln = sdeint.itoint(self.rate_of_experience, self.noise, self.experiences_of_choices, time_vector)
        low_values_indices = soln < 0  # Where values are low
        soln[low_values_indices] = 0

        for i in range(len(self.orbits_for_exp_at_node_1[0])):
            if self.orbits_for_exp_at_node_1[0][i] < 0:
                self.orbits_for_exp_at_node_1[0][i] = 0
            if self.orbits_for_exp_at_node_1[1][i] < 0:
                self.orbits_for_exp_at_node_1[1][i] = 0

        self.plot(soln,t)

    def plot(self, soln,t):
        plt.figure(1)
        plt.subplot(311)
        for i in range(len(self.propotion_of_agents_at_one)):
            plt.plot(range(len(self.propotion_of_agents_at_one[i])), self.propotion_of_agents_at_one[i], label=(chr(97 +i)))
        plt.ylabel('number of agents')
        plt.title('number/experience/utility vs time steps \n NODE_AB')

        plt.subplot(312)
        for i in range(len(self.orbits_for_exp_at_node_1)):
            plt.plot(range(len(self.orbits_for_exp_at_node_1[i])), self.orbits_for_exp_at_node_1[i], label=("Cluster " + str(i)))
        plt.ylabel('experience of agents')

        plt.subplot(313)
        for i in range(len(self.aggregated_utility_node1)):
            plt.plot(range(len(self.aggregated_utility_node1[i])), self.aggregated_utility_node1[i],
                     label=(chr(65 +i)))
        plt.ylabel('utility of choices')
        plt.xlabel('time step')
        plt.legend()

        plt.figure(2)
        plt.subplot(311)
        for i in range(len(self.propotion_of_agents_at_two)):
            plt.plot(range(len(self.propotion_of_agents_at_two[i])), self.propotion_of_agents_at_two[i],
                     label=("choice " + str(i)))
        plt.ylabel('number of agents')
        plt.title('number/experience/utility vs time steps \n NODE_A')

        plt.subplot(312)
        for i in range(self.options):
            plt.plot(t, soln[:, i], label=("choice " + str(i)))
        plt.ylabel('experience of agents')

        plt.subplot(313)
        for i in range(len(self.aggregated_utility_node2)):
            plt.plot(range(len(self.aggregated_utility_node2[i])), self.aggregated_utility_node2[i], label=("A"+str(i+1)))
        plt.ylabel('utility of choices')
        plt.xlabel('time step')
        plt.legend()

        plt.figure(3)
        plt.subplot(311)
        for i in range(len(self.propotion_of_agents_at_three)):
            plt.plot(range(len(self.propotion_of_agents_at_three[i])), self.propotion_of_agents_at_three[i],
                     label=("choice " + str(i+2)))
        plt.ylabel('number of agents')
        plt.title('number/experience/utility vs time steps \n NODE_B')

        plt.subplot(312)
        for i in range(self.options):
            plt.plot(t, soln[:, i+2], label=("choice " + str(i)))
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

    # def return_average_utility(self,time_vector=np.linspace(0, 10, 10000)):
    #     soln = sdeint.itoint(self.rate_of_experience, self.noise, self.experiences_of_choices, time_vector)
    #     return np.average(np.average(self.orbits_utility,0))


def sine_fun(frequency, t, fs, phase):
    y = (np.sin(2 * np.pi * frequency * t/ fs + phase) + 1) / 2
    return y

def main():
    sysd = SystemDynamicsWithSdeint(number_of_agents=100,
                                    k=10,
                                    alpha=2,
                                    utility_of_choices=[0,0,0,0],
                                    initial_experiences=[1, 1, 1, 1],
                                    discount_rate=[0.99,0.99,0.99],
                                    noise_standard_deviation=[0,2,2],
                                    rotation_time=[0,0,0],
                                    rotation_flag=False,
                                    use_fun_for_utilities = True,
                                    frequencies=[50,10,30],
                                    phase=[45,0,0],
                                    utility_flag=False)

    sysd.solve(time_vector=np.linspace(0, 1000, 500))

if __name__ == "__main__":
    main()








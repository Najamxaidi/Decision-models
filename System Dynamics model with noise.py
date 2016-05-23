import numpy as np
import matplotlib.pyplot as plt



class SystemDynamics:
    def __init__(self, agents, discount_rate, alpha, utility_a, utility_b, initial_experience_a, initial_experience_b):
        self.agents = agents
        self.discount_rate = discount_rate  # evaporation rate
        self.alpha = alpha  # model parameter
        self.utility_a = utility_a  # deposit rate for a
        self.utility_b = utility_b # deposit rate for b
        self.experience_a = initial_experience_a   # pheromone level
        self.experience_b = initial_experience_b   # pheromone level
        self.number_of_agents = [[], []]

        # number_of_a = []
        # number_of_b = []
        t = np.linspace(0, 0.001, 20)

        for i in range(len(t)):
            ## initial rate is dependent upon the initial parameters

            rate_a = 0.01 * self.agents * pow(self.experience_a, self.alpha) / (
                pow(self.experience_a, self.alpha) + pow(self.experience_b,self.alpha))
            rate_b = 0.01 * self.agents * pow(self.experience_b, self.alpha) / (
                pow(self.experience_a, self.alpha) + pow(self.experience_b, self.alpha))

            ## agents depends upon the rates
            #number_of_a.append(rate_a * self.agents)
            #number_of_b.append(rate_b * self.agents)

            self.number_of_agents[0].append(rate_a * self.agents)
            self.number_of_agents[1].append(rate_b * self.agents)

            ## calculate new experience
            self.experience_a = self.discount_rate * (self.utility_a * self.number_of_agents[0][i])
            self.experience_b = self.discount_rate * (self.utility_b * self.number_of_agents[1][i])


        ##plotting
        #plt.plot(t, number_of_a, 'g^', t, number_of_b, 'bs')
        #plt.show()

        for i in range(len(self.number_of_agents)):
            plt.plot(t, self.number_of_agents[i], label=i)
        plt.rc('lines', linewidth=2.0)
        plt.legend()
        plt.show()


def main():
    SystemDynamics(agents = 100, discount_rate = 0.5, alpha = 2, utility_a = 5, utility_b = 50,
                       initial_experience_a = 10, initial_experience_b = 10)


if __name__ == "__main__":
    main()



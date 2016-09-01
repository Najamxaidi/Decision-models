import system_dynamics_using_sdeint as systemd
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import numpy as np


class SystemDynamicsStatGenerator:

    def __init__(self,number_of_agents, k, alpha, utility_of_choices, initial_experiences,
                 discount_rate = 0.0, step = 0.01, end_sd = 10, rotation_time = 1000, flag = False):

        self.number_of_agents = number_of_agents
        self.k = k
        self.alpha = alpha
        self.utility_of_choices = np.array(utility_of_choices)
        self.experiences_of_choices = np.array(initial_experiences)
        self.discount_rate = discount_rate
        self.step = step
        self.end_sd = end_sd
        self.rotation_time = rotation_time
        self.flag = flag

    def generate_stat(self, time_vector = np.linspace(0, 1000, 500)):

        area_array = []
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
                                            flag=self.flag
                                            )

            area_array.append(sysd.return_area(time_vector))
            standard_deviation.append(i)

        plt.plot(standard_deviation, area_array, label="test")
        plt.xlabel('standard deviation')
        plt.ylabel('area')
        # plt.legend()
        plt.show()

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
                                                    flag=self.flag
                                                    )

            utility_array.append(sysd.return_average_utility(time_vector))
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

    def fun(self,i):
        sysd = systemd.SystemDynamicsWithSdeint(number_of_agents=self.number_of_agents,
                                                k=self.k,
                                                alpha=self.alpha,
                                                utility_of_choices=self.utility_of_choices,
                                                initial_experiences=self.experiences_of_choices,
                                                discount_rate=self.discount_rate,
                                                sd=i[0],
                                                rotation_time=self.rotation_time,
                                                flag=self.flag
                                                )
        #return -1 * (sysd.return_area(time_vector = np.linspace(0, 1000, 500)))
        return -1 * (sysd.return_average_utility(time_vector=np.linspace(0, 1000, 500)))

    def find_maximum(self, x0):
        minimizer_kwargs = {"method": "BFGS"}
        #x0 = np.array(0.2)
        ret = basinhopping(self.fun, x0, minimizer_kwargs=minimizer_kwargs)
        #ret = differential_evolution(self.fun, x0)
        print("global maximum: x = %.4f, f(x0) = %.4f" % (ret.x, ret.fun))

def main():
    # sdsg = SystemDynamicsStatGenerator(number_of_agents=100,
    #                                 k=1,
    #                                 alpha=2,
    #                                 utility_of_choices=[0.25,0.50],
    #                                 initial_experiences=[0.0001, 0.0001],
    #                                 discount_rate=0.99,
    #                                 step=0.01,
    #                                 end_sd=20,
    #                                 rotation_time=50,
    #                                 flag=True)

    sdsg = SystemDynamicsStatGenerator(number_of_agents=100,
                                    k=1,
                                    alpha=2,
                                    utility_of_choices=[1.5, 3.0],
                                    initial_experiences=[0.001, 0.003],
                                    discount_rate=0.95,
                                    step=0.5,
                                    end_sd=300,
                                    rotation_time=20,
                                    flag=True)

    #sdsg.generate_stat(time_vector = np.linspace(0, 1000, 500))
    sdsg.generate_utility_sweep(time_vector = np.linspace(0, 1000, 500))
    #x0 = np.array(30)
    #x0 = [(0.5, 1.5)]
    #sdsg.find_maximum(x0)

if __name__ == "__main__":
    main()


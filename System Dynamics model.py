


class SystemDynamics:
    def __init__(self):
        print("hello")

    def phi(self, t):
        if t < 60:
            return 1
        elif t >= 60 and t <= 120:
            return 0.5 + (120-t)/120
        else:
            return 0.5



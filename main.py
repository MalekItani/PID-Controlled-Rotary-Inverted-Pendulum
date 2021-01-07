from GenAlgsHelpers import Chromosome, Genetic
import numpy as np
import control
import matplotlib.pyplot as plt


T2V = control.tf([31.3884, 0], [1, 22.1159, -116.116, -514.408])
T1T2 = control.tf([1, 9.02918, -91.4107], [0.79204, 0, 0])


class PIDChromosome(Chromosome):
    @staticmethod
    def evaluate(chrom):
        Kp, Ki, Kd = chrom

        G_c = control.tf([Kp], [1]) + control.tf([Ki], [1, 0]) + control.tf([Kd, 0], [1])

        OLTF = T2V * G_c
        CLTF = control.feedback(OLTF)        
        try:
            stepi = control.step_info(CLTF)
            st = stepi['SettlingTime'] **2 + (1-stepi['SteadyStateValue'])**2
            return st
        except:
            return 1e300
    
    def fitness(chroms):
        fit = np.zeros(len(chroms))
        for i in range(len(chroms)):
            fit[i] = PIDChromosome.evaluate(chroms[i])
        return fit

    def dims():
        return 3
    
    def bounds():
        return np.array([[0,200],
                         [0,200],
                         [0,200]])

def main():
    genetic = Genetic(PIDChromosome, 100, 0.7)

    genetic.run(100)

    best = genetic.best()
    print(best)
    print(PIDChromosome.fitness([best]))

def test():
    Kp, Ki, Kd = [72.94491673, 193.26606581, 2.0693249]
    G_c = control.tf([Kp], [1]) + control.tf([Ki], [1, 0]) + control.tf([Kd, 0], [1])

    OLTF = T2V * G_c
    CLTF = control.feedback(OLTF)

    t = np.linspace(0, 5, 200)
    t, y = control.step_response(CLTF, t)

    print(control.step_info(CLTF, SettlingTimeThreshold=0.05))

    plt.title("Step Response")
    plt.plot(t, y)

    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
    # test()


import numpy as np
import abc
from abc import abstractmethod 
import random

"""
Author: Malek Itani
"""

class Chromosome(metaclass=abc.ABCMeta):
    def __init__(self, dims):
        self.dim = dims
    def compare(self, rhs):
        a = self.fitness(self)
        b = self.fitness(rhs)
        if a < b:
            return -1
        elif a > b:
            return 1
        return 0
    @staticmethod
    @abstractmethod
    def dims():
        pass
    @staticmethod
    @abstractmethod
    def bounds():
        pass
    @abstractmethod
    def fitness(chroms):
        """
        Given n chromosomes, MUST return n real values representing the cost
        """
        pass


class Genetic(object):
    def __init__(self, chromosome_type, n, split, mutation_rate=0.02):
        """
        Defines how a generation should evolve.
        type: Chromosome class, specifies how fitness is computed and dimensions
        of argument of the object function, as well as the bounds for each argument
        n: Number of elements/generation
        split: Proportion of elements to keep per generation (between 0 and 1.0)
        mutation_rate: Probability that a gene mutates after obtaining a new
        generation
        """
        self.chromosome_type = chromosome_type
        self.num_elements = n
        self.removed_per_gen = int((1 - split) * n)
        self.kept_per_gen = n - self.removed_per_gen
        self.mutation_rate = mutation_rate
        if self.removed_per_gen == 1:
            raise Exception("Split invalid, not enough elements removed\
            per generation ({})".format(self.removed_per_gen))
        if self.kept_per_gen == 1:
            raise Exception("Split invalid, not enough elements kept\
            per generation ({})".format(self.kept_per_gen))

    def generate(self):
        bds = self.chromosome_type.bounds()
        self.objects = bds[:,0] + np.random.random((self.num_elements, self.chromosome_type.dims())) * (bds[:,1] - bds[:,0])       

    def evaluate(self):
        return np.argsort(self.chromosome_type.fitness(self.objects))

    def cross(self, order):
        """
        Given the order of the indices of the elements of the current generation,
        generates some new chromosomee from the remaining chromosomes
        """
        i, j = random.sample(range(0, self.kept_per_gen), 2)
        p = order[j]/(order[j] + order[i]) # Probability that a gene is selected from element i
        mask = (np.random.random(self.chromosome_type.dims()) <= p).astype(np.int32)
        return mask * self.objects[order[i]] + (1 - mask) * self.objects[order[j]]

    def evolve(self, order):
        """
        Given the order (in terms of decreasing fitness) of the elements,
        causes the remaining elements per generation to reproduce new
        elements to take the place of the ones that were removed.
        """
        for i in range(self.kept_per_gen, self.num_elements):
            self.objects[order[i]] = self.cross(order)

    def mutate(self):
        """
        Performs a complete mutation of all genes for every chromosome under some probability
        defined by the mutation rate.
        """
        m = self.chromosome_type.dims()
        bds = self.chromosome_type.bounds()
        mutations =  bds[:,0] + np.random.random((self.num_elements, m))\
                                                * (bds[:,1] - bds[:,0]) 
        mask = (np.random.random((self.num_elements, m)) <= self.mutation_rate).astype(np.int32)
        self.objects = mask * mutations + (1 - mask) * self.objects

    def run(self, num_iterations):
        self.generation_count = 1
        self.generate()
        for i in range(num_iterations):
            order = self.evaluate()
            self.evolve(order)
            self.mutate()
            best = self.best()
            best_fitness = self.chromosome_type.fitness([best])
            print("Current best is: {}, fitness is at: {}".format(best, best_fitness))
            self.generation_count += 1
    
    def best(self):
        return self.objects[self.evaluate()[0]]

if __name__ == "__main__":
    class TestChromosome(Chromosome):
        @staticmethod
        def evaluate(chrom):
            return np.linalg.norm(chrom)
        
        def fitness(chroms):
            fit = np.zeros(len(chroms))
            for i in range(len(chroms)):
                fit[i] = TestChromosome.evaluate(chroms[i])
            return fit

        def dims():
            return 3
        
        def bounds():
            return np.array([[0, 100],
                            [0,50],
                            [0,50]])

    genetic = Genetic(TestChromosome, 1000, 0.7)

    genetic.run(100)

    print(genetic.generation_count)
    best = genetic.best()
    print(best)
    print(TestChromosome.fitness([best]))

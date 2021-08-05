import math
from random import random, randint, randrange, uniform

pop_size = 50
crossover_rate = 1
mutation_rate = 0.30
alpha = 4
elite_percent = 0.05
generations = 200

all_features = list()

capacity = 0


class Item:
    def __init__(self, features, class_):
        self.features = features
        self.class_ = class_


def main():
    create_data_struct()
    genetic_algorithm()


def genetic_algorithm():
    genome = generate_genome()
    best_chromosome = []
    for generation in range(generations):

        for chromosome in genome:
            if fitness_function(chromosome) > fitness_function(best_chromosome):
                best_chromosome = chromosome.copy()

        print("current generation: " + str(generation) + "\tbest: " + str(calculate_value(best_chromosome)))

        genome.sort(key=lambda chromo: fitness_function(chromo), reverse=True)

        new_genome = []
        top_chromosomes = math.ceil(elite_percent * len(genome))

        for top in range(top_chromosomes):
            new_genome.append(genome[top])
        while len(new_genome) <= pop_size:
            p1 = list(roulette_wheel_selection(genome))
            p2 = list(roulette_wheel_selection(genome))

            children1, children2 = crossover(p1, p2)

            new_genome.append(mutation(children1, 1, mutation_rate))
            new_genome.append(mutation(children2, 1, mutation_rate))
        genome = new_genome

    return print(calculate_value(best_chromosome))


def create_data_struct():
    lines = list()
    file_name = open("sonar.names", "r")
    for feature in file_name:
        lines.append(feature)
    global all_features
    lines.pop()
    all_features = lines

    file_features = open("sonar.data", "r")
    




def roulette_wheel_selection(genome):
    max_val = sum(fitness_function(chromosome) for chromosome in genome)
    chosen = uniform(0, max_val)
    current = 0
    for chromosome in genome:
        current += fitness_function(chromosome)
        if current > chosen:
            return chromosome


def calculate_weight(chromosome):
    # weight = 0
    # index = 0
    # for bit in chromosome:
    #     if bit == 1:
    #         weight += int(all_items[index].weight)
    #     index += 1
    return 0


def calculate_value(chromosome):
    # value = 0
    # index = 0
    # for bit in chromosome:
    #     if bit == 1:
    #         value += int(all_items[index].value)
    #     index += 1
    return 0


def fitness_function(chromosome):

    return 0


def mutation(chromosome, num: int = 1, probability: float = mutation_rate):
    for x in range(num):
        index = randrange(len(chromosome))
        chromosome[index] = chromosome[index] if random() > probability else abs(chromosome[index] - 1)
    return chromosome


def crossover(chromosome_1, chromosome_2):
    length = len(chromosome_1)
    point = randint(1, length)

    for i in range(point, length):
        chromosome_1[i] = chromosome_2[i]
        chromosome_2[i] = chromosome_1[i]
    return chromosome_1, chromosome_2


def generate_genome():
    genome = list()
    for i in range(pop_size):
        chromosome = list()
        for x in range(len(all_features)):
            chromosome.append(randint(0, 1))
        genome.append(chromosome.copy())

    return genome


if __name__ == '__main__':
    main()

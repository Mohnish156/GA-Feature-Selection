import math
from random import random, randint, randrange, uniform
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.datasets as sk
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

pop_size = 50
crossover_rate = 1
mutation_rate = 0.30
alpha = 4
elite_percent = 0.05
generations = 200
capacity = 0

all_features = list()
sonar = None


class data:
    def __init__(self, x, y, name):
        self.x = x
        self.y = y
        self.name = name


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

        # print("current generation: " + str(generation) + "\tbest: " + str(calculate_value(best_chromosome)))

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

    return print(best_chromosome)


def create_data_struct():
    lines = list()
    file_name = open("sonar.names", "r")
    for feature in file_name:
        lines.append(feature)
    global all_features
    lines.pop()
    all_features = lines

    sonar_data = pd.read_csv("sonar.data")
    wbcd_data = pd.read_csv("wbcd.data")

    print(wbcd_data.head())
    x_data_sonar = sonar_data.iloc[:, :-1].values
    y_data_sonar = sonar_data.iloc[:, 60].values

    x_data_wbcd = wbcd_data.iloc[:, :-1].values
    y_data_wbcd = wbcd_data.iloc[:, 30].values

    global sonar
    sonar = data(x_data_sonar, y_data_sonar, "sonar")
    data_wbcd_formatted = data(x_data_wbcd, y_data_wbcd, "wbcd")


def roulette_wheel_selection(genome):
    max_val = sum(fitness_function(chromosome) for chromosome in genome)
    chosen = uniform(0, max_val)
    current = 0
    for chromosome in genome:
        current += fitness_function(chromosome)
        if current > chosen:
            return chromosome


# https://stackoverflow.com/questions/20297317/python-dataframe-pandas-drop-column-using-int
def fitness_function(chromosome):
    new_data = pd.DataFrame(sonar)
    columns_remove = list()
    for x in range(len(chromosome)):
        if chromosome[x] == 0:
            columns_remove.append(x)

    new_data = new_data.drop(columns=new_data.columns[columns_remove])

    X_train, X_test, y_train, y_test = train_test_split(new_data.x, new_data.y, test_size=0.5)

    scalar = StandardScaler()
    scalar.fit(X_train)
    X_train = scalar.transform(X_train)
    X_test = scalar.transform(X_test)

    classifier = KNeighborsClassifier(n_neighbors=2)
    classifier.fit(X_train, y_train)

    y_prediction = classifier.predict(X_test)

    return  accuracy_score(y_test, y_prediction)


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

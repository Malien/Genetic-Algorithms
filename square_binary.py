from typing import Literal, Optional, TypeVar
import random
import numpy as np


random.seed(42)


def fitness(phenotype: int):
    return phenotype**2


Gene = Literal[0, 1]
Genome = list[Gene]

population_size = 100
genome_length = 10


def encode(phenotype: int) -> Genome:
    gene = bin(int(phenotype * 100))[2:]
    gene = list(map(int, gene))
    padding = [0] * (genome_length - len(gene))
    return padding + gene


def decode(genotype: Genome):
    gene = ''.join(map(str, genotype))
    return int(gene, 2) / 100


def random_genome():
    return [random.randint(0, 1) for _ in range(genome_length)]


def random_population(size):
    return [random_genome() for _ in range(size)]


population = [encode(10.23)] + random_population(population_size - 1)


def crossover(parent1: Genome, parent2: Genome):
    crossover_point = random.randint(1, genome_length - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutate_gene(gene: Gene, rate: float):
    if random.random() < rate:
        return 1 - gene
    return gene


def mutation(genome: Genome, rate: float):
    return [mutate_gene(gene, rate) for gene in genome]


def decide_winner(lhs: Genome, rhs: Genome):
    if fitness(decode(lhs)) > fitness(decode(rhs)):
        return lhs, rhs
    return rhs, lhs


T = TypeVar('T')


def reroll_winner(winner: T, looser: T, chance: float) -> T:
    if random.random() < chance:
        return winner
    else:
        return looser


def stochastic_tournament_selection_with_replacement(
    population: list[Genome],
    chance: float
) -> Genome:
    res = []
    population = population + population
    random.shuffle(population)
    while population:
        lhs = population.pop()
        rhs = population.pop()
        winner, looser = decide_winner(lhs, rhs)
        res.append(reroll_winner(winner, looser, chance))
    return res


def stochastic_tournament_selection_without_replacement(
    population: list[Genome],
    chance: float
) -> Optional[Genome]:
    res = []
    for _ in population:
        # import pdb; pdb.set_trace()
        lhs, rhs = random.sample(population, 2)
        winner, looser = decide_winner(lhs, rhs)
        res.append(reroll_winner(winner, looser, chance))
    return res


iteration_count = 10_000_000


def homougeneusness(population: list[Genome]):
    population = np.array(population, dtype=np.uint8)
    bit_homoneoginity = np.abs(
        np.sum(population, axis=0) / len(population) - 0.5
    ) + 0.5
    return np.min(bit_homoneoginity)


for _ in range(iteration_count):
    population = stochastic_tournament_selection_without_replacement(population, 0.1)
    population = [mutation(genome, 0.01) for genome in population]
    population = [child for child in crossover(*random.sample(population, 2)) for _ in range(population_size // 2)]
    if homougeneusness(population) > 0.9:
        print('Homougeneusness reached 0.9')
        break
    

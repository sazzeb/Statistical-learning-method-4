import random

# Number of individuals in each generation
POPULATION_SIZE = 10

# Length of binary chromosome (5 bits represents 0-31)
GENE_LENGTH = 5

# Probability of mutation per bit
MUTATION_RATE = 0.001


class Individual(object):
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.cal_fitness()

    @classmethod
    def mutated_genes(cls):
        """
        Create random gene (bit) for initialization or mutation
        """
        gene = random.choice([0, 1])
        return gene

    @classmethod
    def create_gnome(cls):
        """
        Create chromosome or string of genes (5-bit binary)
        """
        return [cls.mutated_genes() for _ in range(GENE_LENGTH)]

    def mate(self, par2):
        """
        Perform mating and produce two offspring
        Single-point crossover with random crossover point
        """
        # Choose random crossover point between 1 and GENE_LENGTH-1
        global GENE_LENGTH
        point = random.randint(1, GENE_LENGTH - 1)

        # Create two offspring via crossover
        child1_chromosome = self.chromosome[:point] + par2.chromosome[point:]
        child2_chromosome = par2.chromosome[:point] + self.chromosome[point:]

        # Apply mutation with low probability (0.001 per bit)
        child1_chromosome = self._mutate(child1_chromosome)
        child2_chromosome = self._mutate(child2_chromosome)

        return Individual(child1_chromosome), Individual(child2_chromosome)

    def _mutate(self, chromosome):
        """
        Flip bits with probability MUTATION_RATE
        """
        global MUTATION_RATE, GENE_LENGTH
        new_chromosome = chromosome[:]
        for i in range(GENE_LENGTH):
            if random.random() < MUTATION_RATE:
                # Flip bit: 0 becomes 1, 1 becomes 0
                new_chromosome[i] = 1 - new_chromosome[i]
        return new_chromosome

    def cal_fitness(self):
        """
        Calculate fitness score using f(x) = -x²/10 + 3x
        Higher fitness is better (maximization problem)
        """
        # Convert binary chromosome to integer x
        x = int(''.join(map(str, self.chromosome)), 2)
        
        # Calculate fitness value
        fitness = (-x ** 2) / 10 + 3 * x
        return fitness

    def __str__(self):
        """
        String representation for printing
        """
        x_val = int(''.join(map(str, self.chromosome)), 2)
        chrom_str = ''.join(map(str, self.chromosome))
        return f"{chrom_str} (x={x_val:2d}, f={self.fitness:.4f})"


def select_parent(population, total_fitness):
    """
    Select parent based on fitness proportion (Roulette Wheel Selection)
    P(chromosome i reproduces) = f(x_i) / sum(f(x_k))
    """
    pick = random.uniform(0, total_fitness)
    current = 0
    for individual in population:
        current += individual.fitness
        if current >= pick:
            return individual
    return population[-1]


# Driver code
def main():
    global POPULATION_SIZE

    # Current generation
    generation = 1

    max_generations = 100
    population = []

    # Create initial population
    for _ in range(POPULATION_SIZE):
        gnome = Individual.create_gnome()
        population.append(Individual(gnome))

    print("Maximizing f(x) = -x²/10 + 3x")
    print(f"Population Size: {POPULATION_SIZE}, Chromosome Length: {GENE_LENGTH} bits")
    print(f"Search Space: x ∈ [0, 31], Mutation Rate: {MUTATION_RATE}\n")

    while generation <= max_generations:

        # Sort population by fitness (descending for maximization)
        population = sorted(population, key=lambda x: x.fitness, reverse=True)

        best = population[0]
        avg_fitness = sum(ind.fitness for ind in population) / POPULATION_SIZE

        print("Generation: {:2d}\tBest: {}\tMax: {:.2f}\tAvg: {:.2f}".format(
            generation,
            best,
            best.fitness,
            avg_fitness
        ))

        # Check for convergence (optimal x=15 gives f(x)=22.5)
        if best.fitness >= 22.49:
            print(f"\nOptimal solution found!")
            break

        # Calculate total fitness for selection probabilities
        total_fitness = sum(ind.fitness for ind in population)
        
        if total_fitness == 0:
            break

        # Create new generation
        new_generation = []

        # Perform 5 matings to produce 10 offspring (each mating produces 2)
        for _ in range(POPULATION_SIZE // 2):
            # Select two parents based on fitness proportion
            parent1 = select_parent(population, total_fitness)
            parent2 = select_parent(population, total_fitness)
            
            # Ensure parents are different (optional, for diversity)
            while parent1 == parent2 and POPULATION_SIZE > 1:
                parent2 = select_parent(population, total_fitness)

            # Mate to produce two offspring
            child1, child2 = parent1.mate(parent2)
            new_generation.append(child1)
            new_generation.append(child2)

        population = new_generation
        generation += 1

    # Final result
    population = sorted(population, key=lambda x: x.fitness, reverse=True)
    best_final = population[0]
    x_optimal = int(''.join(map(str, best_final.chromosome)), 2)
    
    print(f"\nFinal Generation: {generation}")
    print(f"Best Chromosome: {best_final}")
    print(f"Optimal x value: {x_optimal} (Expected: 15)")
    print(f"Maximum Fitness: {best_final.fitness:.4f} (Expected: 22.5)")


if __name__ == '__main__':
    main()
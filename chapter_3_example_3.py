"""Chapter 3 binary genetic algorithm example.

This script follows the six-step GA walkthrough from the screenshots:
random initial population, fitness evaluation, selection probabilities,
roulette-wheel parent selection, single-point crossover, and low-rate
mutation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple


POPULATION_SIZE = 10
GENE_LENGTH = 5
MUTATION_RATE = 0.001
SEED = 5215


def bits_to_int(bits: Sequence[int]) -> int:
    return int("".join(str(bit) for bit in bits), 2)


def fitness_function(x: int) -> float:
    return (-x * x / 10.0) + (3.0 * x)


def format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def format_probability(value: float) -> str:
    if abs(value) < 1e-12:
        return "0"
    return f"{value:.4f}"


def selection_weight(fitness: float) -> float:
    return max(0.0, fitness)


def random_bitstring(length: int = GENE_LENGTH) -> str:
    return "".join(str(random.randint(0, 1)) for _ in range(length))


def single_point_crossover(parent1: str, parent2: str, point: int) -> Tuple[str, str]:
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]


def mutate_bitstring(bitstring: str, mutation_rate: float = MUTATION_RATE) -> str:
    bits = list(bitstring)
    for index, bit in enumerate(bits):
        if random.random() < mutation_rate:
            bits[index] = "1" if bit == "0" else "0"
    return "".join(bits)


def roulette_pick(population: Sequence["Individual"], weights: Sequence[float]) -> int:
    total = float(sum(weights))
    if total <= 0:
        return random.randrange(len(population))

    threshold = random.random() * total
    running = 0.0
    for index, weight in enumerate(weights):
        running += float(weight)
        if running >= threshold:
            return index
    return len(population) - 1


@dataclass(frozen=True)
class Individual:
    chromosome: Tuple[int, ...]

    @property
    def bitstring(self) -> str:
        return "".join(str(bit) for bit in self.chromosome)

    @property
    def x_value(self) -> int:
        return bits_to_int(self.chromosome)

    @property
    def fitness(self) -> float:
        return fitness_function(self.x_value)

    @classmethod
    def from_bitstring(cls, bitstring: str) -> "Individual":
        return cls(tuple(int(bit) for bit in bitstring))

    @classmethod
    def create_gnome(cls) -> "Individual":
        return cls.from_bitstring(random_bitstring())

    def mate(self, partner: "Individual") -> Tuple["Individual", "Individual", int]:
        crossover_point = random.randint(1, GENE_LENGTH - 1)
        child_1, child_2 = single_point_crossover(self.bitstring, partner.bitstring, crossover_point)
        child_1 = mutate_bitstring(child_1)
        child_2 = mutate_bitstring(child_2)
        return Individual.from_bitstring(child_1), Individual.from_bitstring(child_2), crossover_point


@dataclass(frozen=True)
class MatingPair:
    parent_1_index: int
    parent_2_index: int
    crossover_point: int | None
    child_1_mutated: bool = False
    child_2_mutated: bool = False

    def expand(self, population: Sequence[Individual]) -> Tuple[str, str, str]:
        parent_1 = population[self.parent_1_index - 1]
        parent_2 = population[self.parent_2_index - 1]

        if self.crossover_point is None:
            return parent_1.bitstring, parent_2.bitstring, f"{parent_1.bitstring}  x  {parent_2.bitstring}"

        child_1, child_2 = single_point_crossover(
            parent_1.bitstring,
            parent_2.bitstring,
            self.crossover_point,
        )
        if self.child_1_mutated:
            child_1 = mutate_bitstring(child_1)
        if self.child_2_mutated:
            child_2 = mutate_bitstring(child_2)
        mating_view = (
            f"{parent_1.bitstring[:self.crossover_point]}|{parent_1.bitstring[self.crossover_point:]}"
            f"  x  {parent_2.bitstring[:self.crossover_point]}|{parent_2.bitstring[self.crossover_point:]}"
        )
        return child_1, child_2, mating_view


def build_table_3_1(population: Sequence[Individual]) -> List[List[str]]:
    fitnesses = [individual.fitness for individual in population]
    weights = [selection_weight(individual.fitness) for individual in population]
    total_weight = sum(weights)
    probs = [weight / total_weight if total_weight > 0 else 1.0 / len(population) for weight in weights]

    rows: List[List[str]] = []
    for chromosome_number, individual in enumerate(population, start=1):
        rows.append(
            [
                str(chromosome_number),
                individual.bitstring,
                str(individual.x_value),
                format_number(individual.fitness),
                format_probability(probs[chromosome_number - 1]),
            ]
        )

    rows.extend(
        [
            ["", "", "Sum", format_number(sum(fitnesses)), format_probability(sum(probs))],
            ["", "", "Average", format_number(sum(fitnesses) / len(fitnesses)), format_probability(sum(probs) / len(probs))],
            ["", "", "Max", format_number(max(fitnesses)), format_probability(max(probs))],
        ]
    )
    return rows


def build_table_3_2(population: Sequence[Individual], mating_pairs: Sequence[MatingPair]) -> List[List[str]]:
    rows: List[List[str]] = []

    for pair in mating_pairs:
        child_1, child_2, mating_view = pair.expand(population)

        for parent_index, child_bitstring in (
            (pair.parent_1_index, child_1),
            (pair.parent_2_index, child_2),
        ):
            x_value = bits_to_int(tuple(int(bit) for bit in child_bitstring))
            rows.append(
                [
                    str(parent_index),
                    mating_view,
                    child_bitstring,
                    str(x_value),
                    format_number(fitness_function(x_value)),
                ]
            )

    fitnesses = [fitness_function(bits_to_int(tuple(int(bit) for bit in row[2]))) for row in rows]
    rows.extend(
        [
            ["", "", "Sum", "", format_number(sum(fitnesses))],
            ["", "", "Average", "", format_number(sum(fitnesses) / len(fitnesses))],
            ["", "", "Max", "", format_number(max(fitnesses))],
        ]
    )
    return rows


def print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))

    def render_row(row: Sequence[str]) -> str:
        return "  ".join(str(cell).ljust(widths[index]) for index, cell in enumerate(row))

    print(render_row(headers))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(render_row(row))


def main() -> None:
    random.seed(SEED)

    # Step 1: generate the initial population randomly.
    population = [Individual.create_gnome() for _ in range(POPULATION_SIZE)]

    print("Step 1: Generate an initial population of 10 random 5-bit chromosomes.")
    print("Step 2: Compute x values and fitness values using f(x) = -x^2/10 + 3x.")
    print("Step 3: Compute selection probabilities from the fitness values.\n")

    print("Table 3.1")
    print_table(
        ["Chromosome Number", "Initial Population", "x Value", "Fitness Value f(x)", "Selection Probability"],
        build_table_3_1(population),
    )

    fitnesses = [person.fitness for person in population]
    weights = [selection_weight(person.fitness) for person in population]

    # Step 4-5: form 5 mating pairs using roulette-wheel selection and random crossover.
    mating_pairs: List[MatingPair] = []
    for _ in range(POPULATION_SIZE // 2):
        parent_1_index = roulette_pick(population, weights)
        parent_2_index = roulette_pick(population, weights)
        while parent_2_index == parent_1_index and len(population) > 1:
            parent_2_index = roulette_pick(population, weights)

        parent_1 = population[parent_1_index]
        parent_2 = population[parent_2_index]
        child_1, child_2, crossover_point = parent_1.mate(parent_2)
        mating_pairs.append(
            MatingPair(
                parent_1_index=parent_1_index + 1,
                parent_2_index=parent_2_index + 1,
                crossover_point=crossover_point,
                child_1_mutated=child_1.bitstring != single_point_crossover(parent_1.bitstring, parent_2.bitstring, crossover_point)[0],
                child_2_mutated=child_2.bitstring != single_point_crossover(parent_1.bitstring, parent_2.bitstring, crossover_point)[1],
            )
        )

    print("\nStep 4: Select chromosomes for 5 matings using fitness-based probabilities.")
    print("Step 5: Apply single-point crossover and low-probability mutation to the offspring.")
    print("Step 6: Evaluate the new population and display Table 3.2.\n")

    print("Table 3.2")
    print_table(
        ["Chromosome Number", "Mating Pairs", "New Population", "x Value", "Fitness Value f(x)"],
        build_table_3_2(population, mating_pairs),
    )


if __name__ == "__main__":
    main()
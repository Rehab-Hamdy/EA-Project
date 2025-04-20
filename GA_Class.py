import numpy as np
import random


class Genetic_Algorithm:
    def __init__(self, weights):
        self.weights = np.array(weights, dtype=float)  # NN weights (x₁, ..., xₙ)
        self.sigmas = np.ones_like(weights) * 0.01  # Initialize all σᵢ to 0.01
        self.fitness = None


    # ___ Random Initialization ___ #
    @staticmethod
    def initialize_population(pop_size, nn_architecture):
        """
        pop_size: int, number of individuals in population
        nn_architecture: list of tuples (input_size, output_size) per layer
        """
        population = []
        for _ in range(pop_size):
            weights = []
            for (n_in, n_out) in nn_architecture:
                # Xavier Glorot normal initialization
                scale = np.sqrt(2 / (n_in + n_out)) # Xavier normal
                layer_weights = np.random.randn(n_in * n_out) * scale 
                weights.extend(layer_weights)
            individual = Genetic_Algorithm(weights)
            population.append(individual)
        return population


    # ___ Parent Selection ___ #
    @staticmethod
    def tournament_selection(population, fitnesses, num_parents, tournament_size):
        """
        Performs tournament selection to choose parents.

        Args:
            population (list): List of GeneticAlgorithm individuals.
            fitnesses (list): List of corresponding fitness values.
            num_parents (int): Number of individuals to select.
            tournament_size (int): Number of candidates per tournament.

        Returns:
            list: Selected individuals.
        """
        mating_pool = []
        population_size = len(population) # μ

        for _ in range(num_parents): # num_parents => λ (lambda)
            # Select k unique individuals without replacement
            candidates_indices = random.sample(range(population_size), tournament_size)
            candidates_fitness = [fitnesses[i] for i in candidates_indices]

            # Select the best among them (minimization or maximization problem)
            best_index = candidates_indices[candidates_fitness.index(max(candidates_fitness))]  # Change to `min` if minimizing
            mating_pool.append(population[best_index])

        return mating_pool


    # ___ Crossover (Recombination) ___ #
        """
        Assume -> x_i < y_i
        difference d_i = y_i - x_i
        the range for the i-th value in the child z -> z_i = [x_i - αd , x_i + αd] => lower_bound & upper_bound
        To create a child, we can sample a random number u uniformly from [0,1]
        Calc γ = (1 - 2α)u - α and Set z_i = (1 - γ)x_i + γ * y_i

        The best practice for α = 0.5
        """
    @staticmethod
    def BLX_alpha_crossover(parent1, parent2, alpha=0.5): # Blend Crossover
        
        # Ensure parents are numpy arrays and whether they are of the same length
        parent1 = np.array(parent1, dtype=float)
        parent2 = np.array(parent2, dtype=float)

        if parent1.ndim != 1 or parent2.ndim != 1:
            raise ValueError("Parents must be 1-dimensional arrays")
        if len(parent1) != len(parent2):
            raise ValueError("Parents must have the same length")
        
        # Step 1: Ensure X < Y for each coordinate
        x = np.minimum(parent1, parent2)
        y = np.maximum(parent1, parent2)
        
        # Step 2: Compute the difference d_i = y_i - x_i
        d = y - x
        
        # Step 3: Compute the range for the child z_i = [x_i - αd_i, x_i + αd_i]
        lower_bound = x - alpha * d
        upper_bound = y + alpha * d
        
        # Step 4: Sample a random number u uniformly from [0, 1[
        u = np.random.uniform(0, 1, size=len(parent1)) # Generate a vector of random numbers uniformly distributed in the range [0, 1), with the same length as parent1
        
        # Step 5: Calculate γ = (1 - 2α)u - α
        gamma = (1 - 2 * alpha) * u - alpha
        
        # Step 6: Compute the child z_i = (1 - γ)x_i + γy_i
        child = (1 - gamma) * x + gamma * y

        if np.all(child <= upper_bound) and np.all(child >= lower_bound):
            return child
        else:
            return "The child is out of the boundaries"


    # ___ Mutation ___ #
    @staticmethod
    def mutate_uncorrelated_n_step(individual, tau, tau_prime, epsilon=1e-10): # Self-adaptive mutation
        n = len(individual.weights)
        
        # Global and local noise
        global_noise = np.random.normal(0, 1)
        local_noise = np.random.normal(0, 1, n)
        
        # Update σᵢ (Eq. 4.4)
        new_sigmas = individual.sigmas * np.exp(tau_prime * global_noise + tau * local_noise)
        
        # Apply boundary rule to prevent σ ≈ 0 (from your image)
        new_sigmas = np.clip(new_sigmas, epsilon, None)
        
        # Mutate weights (Eq. 4.5)
        new_weights = individual.weights + new_sigmas * np.random.normal(0, 1, n)
        
        # Return new individual
        mutated = Genetic_Algorithm(new_weights)
        mutated.sigmas = new_sigmas
        return mutated
    

    # ___ Survivor Selection (Elitism + GENITOR) ___ #
    @staticmethod
    def hybrid_survivor_selection(population, offspring, elitism_count=1, genitor_count=None):
        """
        Combines Elitism and GENITOR (Replace-Worst) strategies.
        
        Args:
            population (list): Current generation (list of individuals with .fitness attribute).
            offspring (list): New individuals generated from crossover + mutation.
            elitism_count (int): Number of best individuals to carry over from current generation.
            genitor_count (int or None): Number of worst individuals to replace. If None, auto-calculated.

        Returns:
            list: Next generation.
        """
        # Ensure all individuals have a defined fitness
        assert all(ind.fitness is not None for ind in population), "Current population must have evaluated fitness"
        assert all(ind.fitness is not None for ind in offspring), "Offspring must have evaluated fitness"

        # Sort current population by fitness (descending: assuming maximization)
        sorted_population = sorted(population, key=lambda ind: ind.fitness, reverse=True)

        # Elitism: Keep top individuals
        elites = sorted_population[:elitism_count]

        # GENITOR: Replace the worst individuals with best offspring
        if genitor_count is None:
            genitor_count = len(offspring) - elitism_count
        
        sorted_offspring = sorted(offspring, key=lambda ind: ind.fitness, reverse=True)
        top_offspring = sorted_offspring[:genitor_count]

        # Replace worst individuals
        survivors = elites + top_offspring

        # Fill remaining if needed
        total_needed = len(population)
        if len(survivors) < total_needed:
            remaining = sorted_population[elitism_count:-genitor_count] if genitor_count > 0 else sorted_population[elitism_count:]
            survivors.extend(remaining[:total_needed - len(survivors)])

        # Truncate if too long
        return survivors[:total_needed]
    

import numpy as np

class Individual:
    def __init__(self, weights):
        self.weights = np.array(weights)  # NN weights (x₁, ..., xₙ)
        self.sigmas = np.ones_like(weights) * 0.01  # Initialize all σᵢ to 0.01
        self.fitness = None


    def mutate_uncorrelated_n_step(individual, tau, tau_prime, epsilon=1e-10):
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
        mutated = Individual(new_weights)
        mutated.sigmas = new_sigmas
        return mutated

  """
    Assume -> x_i < y_i
    difference d_i = y_i - x_i
    the range for the i-th value in the child z -> z_i = [x_i - αd , x_i + αd] => lower_bound & upper_bound
    To create a child, we can sample a random number u uniformly from [0,1]
    Calc γ = (1 - 2α)u - α and Set z_i = (1 - γ)x_i + γ * y_i

    The best practice for α = 0.5
"""

    def BLX_alpha(parent1, parent2, alpha=0.5): # Blend Crossover
    
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

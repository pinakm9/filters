import filter as fl 
import  numpy as np
import copy

class GeneticPF(fl.ParticleFilter):

    def __init__(self, model, particle_count, record_path = None, particles = None, generations_per_step=3,\
                 mutation_prob=0.2, ellitism_factor=0.5, max_population=2000, mutation_size=0.01):
        super().__init__(model=model, particle_count=particle_count, record_path=record_path, particles=particles)
        self.generations_per_step = generations_per_step
        self.mutation_prob = mutation_prob 
        self.ellitism_factor = ellitism_factor
        self.max_population = max_population
        self.mutation_size = mutation_size
        self.cov = 0.1 * np.identity(self.dimension)

    def crossover_regular(self, particle_m, particle_d):
        # find a cross-over point
        alpha = np.random.randint(self.dimension)
        beta_1, beta_2 = np.random.uniform(size=2)
        offsprings = [np.zeros(self.dimension) for i in range(4)]
        for j in range(self.dimension):
            if j < alpha:
                offsprings[0][j] = particle_m[j]
                offsprings[1][j] = particle_d[j]
                offsprings[2][j] = particle_m[j]
                offsprings[3][j] = particle_d[j]

            elif  j >= alpha:
                offsprings[0][j] = particle_m[j] - beta_1 * (particle_m[j] - particle_d[j])
                offsprings[1][j] = particle_d[j] + beta_1 * (particle_m[j] - particle_d[j])
                offsprings[2][j] = particle_m[j] - beta_2 * (particle_m[j] - particle_d[j])
                offsprings[3][j] = particle_d[j] + beta_2 * (particle_m[j] - particle_d[j])
        return offsprings + [particle_m, particle_d]

    def crossover(self, particle_m, particle_d):
        # find a cross-over point
        beta = np.random.uniform(size=4)
        offsprings = []
        for i in range(4):
             mu = beta[i] * particle_m + (1.0 - beta[i]) * particle_d
             offsprings.append(np.random.multivariate_normal(mu, self.cov))
        return offsprings + [particle_m, particle_d]

    def mutate(self):
        # find particles to be mutated
        idx = np.random.choice(2, p=[1.0-self.mutation_prob, self.mutation_prob], size=len(self.current_population))
        for i in idx:
            alpha = np.random.randint(self.dimension)
            self.current_population[i][alpha] += np.random.normal(scale=self.mutation_size)

    def breed_first_gen(self):
        # create a mating pool
        num_mating_pairs = int(self.max_population / 6)
        idx_m = list(range(num_mating_pairs))#np.random.choice(self.particle_count, size=num_mating_pairs)
        idx_d = np.random.choice(self.particle_count, size=num_mating_pairs)
        for p in range(num_mating_pairs):
            if idx_m[p] == idx_m[p]:
                if idx_m[p] < self.particle_count - 1:
                    idx_m[p] += 1
                else:
                    idx_m[p] -= 1
        # breed
        self.current_population = []
        for p in range(num_mating_pairs):
            offsprings = self.crossover(self.particles[idx_m[p]], self.particles[idx_d[p]])
            self.current_population += [self.model.hidden_state.sims[self.current_time].algorithm(self.current_time, particle)\
                                         for particle in offsprings]
        self.mutate()

    def breed_later_gen(self):
        # create a mating pool
        num_mating_pairs = int(self.max_population / 6)
        idx_m = list(range(num_mating_pairs))#np.random.choice(self.particle_count, size=num_mating_pairs)
        idx_d = np.random.choice(self.particle_count, size=num_mating_pairs)
        for p in range(num_mating_pairs):
            if idx_m[p] == idx_m[p]:
                if idx_m[p] < self.particle_count - 1:
                    idx_m[p] += 1
                else:
                    idx_m[p] -= 1
        # breed
        self.current_population = []
        for p in range(num_mating_pairs):
            self.current_population += self.crossover(self.particles[idx_m[p]], self.particles[idx_d[p]])
         
        self.mutate()

    def select(self, observation):
        self.weights = [self.model.observation.conditional_pdf(self.current_time, observation, particle)\
                        for particle in self.current_population]
        idx = np.argsort(self.weights)
        self.particles = np.array(self.current_population)[idx][::-1][:self.particle_count]
        self.weights = np.array(self.weights)[idx][::-1][:self.particle_count]

    def one_step_update(self, observation, particles = None):
        """
        Description:
            Updates weights according to the last observation
        Args:
            observation: an observation of dimension = self.dimension
        Returns:
            self.weights
        """
        # predict the new particles
        if self.current_time == 0 and len(self.particles) != self.particle_count:
            self.particles = self.model.hidden_state.sims[0].generate(self.particle_count)

        self.breed_first_gen()
        self.select(observation)
        for _ in range(self.generations_per_step):
            self.breed_later_gen()
            self.select(observation)

        # normalize weights
        print('step: {}, sum of weights: {}'.format(self.current_time, self.weights.sum()))
        self.weights /= self.weights.sum()
        if np.isnan(self.weights[0]) or np.isinf(self.weights[0]):
            self.status = 'faliure'


    def update(self, observations, threshold_factor = 0.1, method = 'mean', resampling_method = 'systematic', record_path = None, **params):
        """
        Description:
            Updates using all the obeservations using self.one_step_update and self.resample
        Args:
            observations: list/np.array of observations to pass to self.one_step_update
            threshold_factor: fraction of self.particle_count which the effective particle count has to surpass in order to stop resampling
            method: method for computing trajectory, default = 'mean'
            resampling method: method for resampling, default = 'systematic'
            record_path: file path for storing evolution of particles
        Returns:
            self.weights
        """
        self.observed_path = observations
        for observation in self.observed_path:
            self.one_step_update(observation = observation)
            self.resample(threshold_factor = 0.0, method = resampling_method, **params)
            if method is not None:
                self.compute_trajectory(method = method)
            self.record(observation)
            self.current_time += 1
        if not hasattr(self, 'status'):
            self.status = 'success'
        return self.status
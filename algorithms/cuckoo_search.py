import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.special import gamma
from collections import defaultdict
from algorithms.differential_evolution import DifferentialEvolution

class DiscreteCuckooSearch:
    """
    Implements Algorithm 2: Discrete Cuckoo Search Based on Differential Evolution
    """
    
    def __init__(self, config, network, tasks):
        self.config = config
        self.network = network
        self.tasks = tasks
        
        self.cv_targets = list(network.collaborative_vehicles)
        self.ap_targets = list(network.access_points)
        self.offload_targets = self.cv_targets + self.ap_targets 
        self.num_targets = len(self.offload_targets)
        
        self.nest_size = config['algorithms']['cuckoo']['nest_size']
        self.max_iterations = config['algorithms']['cuckoo']['max_iterations']
        self.pa = config['algorithms']['cuckoo'].get('discovery_probability', 0.25)
        self.alpha = config['algorithms']['cuckoo'].get('step_size_factor', 1.0)
        self.beta = config['algorithms']['cuckoo']['levy_beta']
        
        self.scaling_factor = config['algorithms']['differential_evolution']['scaling_factor']
        self.crossover_prob = config['algorithms']['differential_evolution']['crossover_probability']
        
        self.dimension = len(tasks) if tasks else 0
        
        self.nests = None
        self.fitness = None
        self.best_nest = None
        self.best_fitness = float('inf')
        
        self.source_vehicle_map = {}
        for mv in self.network.mission_vehicles:
            if mv.current_task:
                self.source_vehicle_map[mv.current_task.id] = mv
                
        self.node_resources = {}
        for target in self.offload_targets:
            self.node_resources[target.id] = target.computing_resources

        self.de = DifferentialEvolution(
            scaling_factor=self.scaling_factor,
            crossover_prob=self.crossover_prob
        )

    def initialize_nests(self):
        if self.dimension == 0:
            self.nests, self.fitness, self.best_nest = [], [], []
            return
            
        self.nests = np.random.rand(self.nest_size, self.dimension)
        self.fitness = np.full(self.nest_size, float('inf'))
        
        for i in range(self.nest_size):
            self.fitness[i] = self.evaluate_fitness(self.nests[i])
            
        if self.fitness.size > 0:
            best_idx = np.nanargmin(self.fitness)
            if not np.isnan(self.fitness[best_idx]):
                self.best_nest = self.nests[best_idx].copy()
                self.best_fitness = self.fitness[best_idx]
            else:
                self.best_nest = self.nests[0].copy()
                self.best_fitness = 10000.0 # 10,000 ms
        else:
            self.best_fitness = 10000.0
        
    def levy_flight(self):
        if self.dimension == 0: return np.array([])
        sigma_u = (gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2) / 
                  (gamma((1 + self.beta) / 2) * self.beta * 2**((self.beta - 1) / 2)))**(1 / self.beta)
        sigma_v = 1
        u = np.random.normal(0, sigma_u, self.dimension)
        v = np.random.normal(0, sigma_v, self.dimension)
        step = 0.01 * u / (np.abs(v)**(1 / self.beta) + 1e-10)
        return step
    
    def continuous_to_discrete(self, continuous_solution):
        """Maps a continuous value [0, 1] to a discrete target node index."""
        if self.num_targets == 0:
            return np.array([], dtype=int)
            
        discrete_solution = np.zeros(len(continuous_solution), dtype=int)
        for i, value in enumerate(continuous_solution):
            target_index = int(value * self.num_targets)
            if target_index >= self.num_targets:
                target_index = self.num_targets - 1
            discrete_solution[i] = target_index
        return discrete_solution
    
    def evaluate_fitness(self, solution):
        """
        Evaluate fitness (average task processing delay in MILLISECONDS)
        """
        if not self.tasks or self.dimension == 0 or self.num_targets == 0:
            return 0.0
            
        discrete_solution = self.continuous_to_discrete(solution)
        total_delay_ms = 0
        
        node_load_count = defaultdict(int)
        task_assignments = {} 
        
        best_ap_map = {}
        for task in self.tasks:
            best_ap_map[task.id] = self._find_best_ap_target(task)
        
        for i, task in enumerate(self.tasks):
            source_vehicle = self._get_source_vehicle(task)
            if not source_vehicle:
                task_assignments[task.id] = (None, (10000.0, 0)) # 10s penalty
                continue
                
            target_index = discrete_solution[i]
            target_node = self.offload_targets[target_index]
            
            final_node = None
            delay_components = (0, 0) # (transmission_delay, processing_delay_base)
            
            if target_node in self.cv_targets:
                cv_is_valid = False
                if task.required_service in target_node.cached_services:
                    distance = source_vehicle.distance_to(target_node)
                    if distance <= source_vehicle.communication_range: # Check MV's range
                        cv_is_valid = True
                
                if cv_is_valid:
                    final_node = target_node
                    tup = self._calculate_transmission_delay_ms(
                        task.data_size, distance, self.network.config['vehicle_tx_power_dbm']
                    )
                    tul_base = task.computation_demand # MCycles
                    delay_components = (tup, tul_base)
            
            if final_node is None:
                ap_node, trans_delay, proc_base, is_forwarded = best_ap_map[task.id]

                if ap_node is None:
                    task_assignments[task.id] = (None, (10000.0, 0))
                else:
                    final_node = ap_node
                    delay_components = (trans_delay, proc_base)

            if final_node:
                node_load_count[final_node.id] += 1
                task_assignments[task.id] = (final_node, delay_components)
            elif task.id not in task_assignments:
                task_assignments[task.id] = (None, (10000.0, 0))

        for i, task in enumerate(self.tasks):
            final_node, (transmission_delay, processing_base) = task_assignments.get(task.id, (None, (10000.0, 0)))
            
            if final_node is None:
                total_delay_ms += transmission_delay # 10,000ms
                continue
            
            node_id = final_node.id
            node_total_resource = self.node_resources.get(node_id, 0.1)
            load = node_load_count[node_id]
            if load == 0: load = 1
                
            allocated_resource = node_total_resource / load
            
            processing_delay = self._calculate_processing_delay_ms(
                processing_base, allocated_resource
            )
            
            total_delay_ms += (transmission_delay + processing_delay)
            
        return total_delay_ms / len(self.tasks) if self.tasks else 0.0

    def _get_source_vehicle(self, task):
        return self.source_vehicle_map.get(task.id)

    def _calculate_transmission_delay_ms(self, data_size_kbytes: float, distance_m: float, tx_power_dbm: float, is_fiber=False):
        data_size_mbits = data_size_kbytes * 8 / 1000.0 # KBytes to Mbits
        if is_fiber:
            delay_sec = data_size_mbits / (1000 + 1e-6) # 1 Gbps
            return delay_sec * 1000
        capacity_mbps = self.network.get_channel_capacity_mbps(distance_m, tx_power_dbm)
        if capacity_mbps <= 0.1: return 10000.0 
        delay_sec = data_size_mbits / capacity_mbps
        return delay_sec * 1000

    def _calculate_processing_delay_ms(self, task_mcycles: float, allocated_resources_ghz: float):
        if allocated_resources_ghz <= 0:
            return 10000.0
        delay_sec = task_mcycles / (allocated_resources_ghz * 1000)
        return delay_sec * 1000

    def _find_associated_ap(self, task):
        """Finds the closest AP in the MV's range for a task."""
        source_vehicle = self._get_source_vehicle(task)
        if not source_vehicle: return None, float('inf')
        
        associated_ap = None
        min_dist = float('inf')
        
        for ap in self.ap_targets:
            distance = ap.distance_to(source_vehicle)
            # ---
            # --- THIS IS THE CRITICAL BUG FIX ---
            # ---
            # Check the MV's communication range (100m)
            if distance < min_dist and distance <= source_vehicle.communication_range:
                min_dist = distance
                associated_ap = ap
        
        return associated_ap, min_dist
        
    def _find_best_ap_target(self, task):
        """
        Finds the best AP target (associated or forwarded) for a task.
        Returns (node, trans_delay, proc_base, is_forwarded)
        """
        associated_ap, assoc_dist = self._find_associated_ap(task)
        
        if associated_ap is None:
            return None, 10000.0, 0, False 

        tup_m_k = self._calculate_transmission_delay_ms(
            task.data_size, assoc_dist, self.network.config['ap_tx_power_dbm']
        )
        tul_base = task.computation_demand # This is MCycles
        
        delay_assoc = tup_m_k + self._calculate_processing_delay_ms(tul_base, associated_ap.computing_resources)
        
        delay_fwd = float('inf')
        best_fwd_ap = None
        ts_k_k_prime = self._calculate_transmission_delay_ms(task.data_size, 0, 0, is_fiber=True)

        for ap in self.ap_targets:
            if ap.id == associated_ap.id:
                continue
            
            fwd_proc_delay = self._calculate_processing_delay_ms(tul_base, ap.computing_resources)
            total_fwd_delay = tup_m_k + ts_k_k_prime + fwd_proc_delay
            
            if total_fwd_delay < delay_fwd:
                delay_fwd = total_fwd_delay
                best_fwd_ap = ap
                
        if delay_assoc <= delay_fwd or best_fwd_ap is None:
            return associated_ap, tup_m_k, tul_base, False
        else:
            return best_fwd_ap, (tup_m_k + ts_k_k_prime), tul_base, True


    def _run_differential_evolution(self, i):
        """Implements DE (Eq. 22-24) for local search"""
        mutant = self.de.mutation(self.nests, i)
        trial = self.de.crossover(self.nests[i], mutant)
        trial_fitness = self.evaluate_fitness(trial)
        
        if np.isnan(trial_fitness):
            trial_fitness = 10000.0
            
        if trial_fitness < self.fitness[i]:
            self.nests[i] = trial
            self.fitness[i] = trial_fitness
            
    def run(self):
        """Executes Algorithm 2"""
        if not self.tasks or self.dimension == 0 or self.num_targets == 0:
            return np.array([]), 0.0, []
            
        self.initialize_nests()
        
        fitness_history = []
        
        if self.best_fitness == float('inf'):
             fitness_history.append(100) # 100ms
        elif np.isnan(self.best_fitness):
             fitness_history.append(100)
        else:
             fitness_history.append(self.best_fitness)
        
        for iteration in range(self.max_iterations):
            if self.best_nest is not None and len(self.best_nest) == self.dimension:
                for i in range(self.nest_size):
                    step = self.levy_flight()
                    if step.size == 0: continue
                    new_nest = self.nests[i] + self.alpha * step * (self.best_nest - self.nests[i])
                    new_nest = np.clip(new_nest, 0, 1)
                    
                    new_fitness = self.evaluate_fitness(new_nest)
                    
                    if new_fitness < self.fitness[i]:
                        self.nests[i] = new_nest
                        self.fitness[i] = new_fitness
                        if new_fitness < self.best_fitness:
                            self.best_nest = new_nest.copy()
                            self.best_fitness = new_fitness
            
            for i in range(self.nest_size):
                if np.random.random() < self.pa:
                    self._run_differential_evolution(i)
            
            if np.isnan(self.best_fitness) or self.best_fitness == float('inf'):
                fitness_history.append(fitness_history[-1]) 
            else:
                fitness_history.append(self.best_fitness)
                    
        return self.continuous_to_discrete(self.best_nest), self.best_fitness, fitness_history
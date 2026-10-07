import matplotlib.pyplot as plt
import numpy as np
from simulation.simulator import ECORASimulator
import traceback
import copy
import time

def run_experiment(config_file: str, sim_params: dict):
    """
    Runs a single simulation and returns the convergence history (in MILLISECONDS)
    and the final AP load distribution.
    """
    print(f"--- Running experiment with params: {sim_params} ---")
    
    simulator = ECORASimulator(config_file, sim_params)
    simulator.initialize_network()
    
    # Run 10 trials (time slots) and average results
    all_histories = []
    all_ap_loads = []
    
    print("  > Running 10 trials (time slots) to average results...")
    for i in range(10):
        simulator.current_time = i
        history, ap_loads = simulator.run_time_slot(return_loads=(i == 9)) 
        if history: # If tasks were generated
            all_histories.append(history)
        if i == 9:
            all_ap_loads = ap_loads

    if not all_histories:
        print("--- Experiment finished. No tasks generated in 10 trials. ---")
        max_iter = simulator.config['algorithms']['cuckoo']['max_iterations']
        return [100.0] * (max_iter + 1), {} # Return 100ms penalty

    # Average the convergence histories
    avg_history = np.mean(np.array(all_histories), axis=0)
    
    max_iter = simulator.config['algorithms']['cuckoo']['max_iterations']
    if len(avg_history) < (max_iter + 1):
        last_val = avg_history[-1]
        padding = [last_val] * ((max_iter + 1) - len(avg_history))
        avg_history = np.concatenate((avg_history, padding))

    print(f"--- Experiment finished. Avg final delay: {avg_history[-1]:.4f} ms ---")
    return avg_history, all_ap_loads


def plot_figure_convergence(results: dict, title: str, filename: str):
    """
    Generic plotting function for convergence graphs (Figs 2, 3, 4)
    """
    plt.figure(figsize=(10, 6))
    x_ticks_paper = np.arange(0, 26, 5) # 0, 5, 10, 15, 20, 25
    
    for label, history_ms in results.items():
        if len(history_ms) > 0:
            # Plot directly in ms
            plt.plot(history_ms, label=f"ECORA ({label})", marker='o', markersize=4, linestyle='-') 
        
    plt.title(title)
    plt.xlabel('The number of iterations(times)')
    plt.ylabel('Average task processing latency (ms)')
    plt.legend()
    plt.grid(True)
    plt.xticks(x_ticks_paper)
    plt.xlim(0, 25)
    plt.savefig(filename)
    print(f"\nSaved Figure to '{filename}'")
    plt.show()

def plot_figure_5_load_balance(ap_loads: dict, filename: str):
    """
    Plots the AP load distribution (Fig 5)
    """
    if not ap_loads:
        print("Could not generate Fig 5: No AP loads recorded.")
        return
        
    plt.figure(figsize=(8, 6))
    
    ap_ids = sorted(ap_loads.keys())
    loads = [ap_loads.get(ap_id, 0) for ap_id in ap_ids] # Use .get for safety
    labels = [f"AP {i+1}" for i in range(len(ap_ids))] # Use 1-based index

    plt.bar(labels, loads, color='c', label='ECORA (Our Sim)') # Use 'c' (cyan) like paper
    
    plt.title('Fig 5 (ECORA): AP Load Distribution')
    plt.xlabel('AP ID')
    plt.ylabel('AP Load (Number of Tasks)')
    plt.legend()
    plt.grid(axis='y')
    plt.savefig(filename)
    print(f"\nSaved Figure to '{filename}'")
    plt.show()


if __name__ == "__main__":
    CONFIG_FILE = 'config.yaml'
    start_time = time.time()
    
    try:
        # ---
        # --- Figure 2: Varying Traffic Density ---
        # ---
        print("\n" + "="*50)
        print("RUNNING FIGURE 2 EXPERIMENT (TRAFFIC DENSITY)")
        print("="*50)
        traffic_densities = [0.06, 0.08, 0.10, 0.12]
        fig_2_results = {}
        
        final_ap_loads = {}
        
        for density in traffic_densities:
            params = {
                'task_generation_prob': density,
                'average_speed_kmh': 60 
            }
            history, ap_loads = run_experiment(CONFIG_FILE, params)
            fig_2_results[f"Density={density}"] = history
            if density == 0.10: # Store the loads for Fig 5
                final_ap_loads = ap_loads
            
        plot_figure_convergence(
            fig_2_results,
            'Fig 2 (ECORA): Avg. Delay vs. Iterations (Varying Traffic Density)',
            'ecora_figure_2_traffic_density.png'
        )

        # ---
        # --- Figure 3: Varying Average Speed ---
        # ---
        print("\n" + "="*50)
        print("RUNNING FIGURE 3 EXPERIMENT (AVERAGE SPEED)")
        print("="*50)
        average_speeds = [30, 60, 90, 120]
        fig_3_results = {}
        
        for speed in average_speeds:
            params = {
                'task_generation_prob': 0.10, # Use a constant density
                'average_speed_kmh': speed
            }
            history, _ = run_experiment(CONFIG_FILE, params)
            fig_3_results[f"Speed={speed} km/h"] = history
            
        plot_figure_convergence(
            fig_3_results,
            'Fig 3 (ECORA): Avg. Delay vs. Iterations (Varying Average Speed)',
            'ecora_figure_3_average_speed.png'
        )

        # ---
        # --- Figure 4: Varying Algorithm Parameters (Pa and alpha3) ---
        # ---
        print("\n" + "="*50)
        print("RUNNING FIGURE 4a EXPERIMENT (Varying Pa)")
        print("="*50)
        pa_values = [0.05, 0.15, 0.25, 0.35, 0.45]
        fig_4a_results = {}
        
        base_params_fig4 = {
            'task_generation_prob': 0.10,
            'average_speed_kmh': 60
        }
        
        for pa in pa_values:
            params = copy.deepcopy(base_params_fig4)
            params['discovery_probability'] = pa 
            
            history, _ = run_experiment(CONFIG_FILE, params)
            fig_4a_results[f"Pa={pa}"] = history
            
        plot_figure_convergence(
            fig_4a_results,
            'Fig 4a (ECORA): Avg. Delay vs. Iterations (Varying Pa)',
            'ecora_figure_4a_pa.png'
        )
        
        print("\n" + "="*50)
        print("RUNNING FIGURE 4b EXPERIMENT (Varying alpha3)")
        print("="*50)
        alpha3_values = [0.01, 0.1, 1, 5, 10]
        fig_4b_results = {}
        
        for a3 in alpha3_values:
            params = copy.deepcopy(base_params_fig4)
            params['step_size_factor'] = a3 
            
            history, _ = run_experiment(CONFIG_FILE, params)
            fig_4b_results[f"a3={a3}"] = history
            
        plot_figure_convergence(
            fig_4b_results,
            'Fig 4b (ECORA): Avg. Delay vs. Iterations (Varying a3)',
            'ecora_figure_4b_alpha3.png'
        )
        
        # ---
        # --- Figure 5: Load Balancing ---
        # ---
        print("\n" + "="*50)
        print("PLOTTING FIGURE 5 EXPERIMENT (LOAD BALANCING)")
        print("="*50)
        plot_figure_5_load_balance(
            final_ap_loads, # Use the loads from the Fig 2 run
            'ecora_figure_5_load_balance.png'
        )

    except Exception as e:
        print(f"Error during simulation: {e}")
        traceback.print_exc()
        
    print(f"\nTotal simulation time: {time.time() - start_time:.2f} seconds")
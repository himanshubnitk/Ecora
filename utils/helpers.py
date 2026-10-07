import numpy as np
# from scipy.stats import zipf # <- This was the bug

def generate_zipf_popularity(num_services, alpha):
    """
    Generate service popularity following Zipf distribution
    This manually implements the paper's Equation 1:
    p(f) = (f**-alpha) / sum(k**-alpha for k in 1..F)
    """
    if num_services <= 0:
        return np.array([])
        
    # Create an array for k = 1, 2, ..., num_services
    ranks = np.arange(1, num_services + 1)
    
    # Calculate p(f) = f**-alpha
    probabilities = ranks**(-alpha)
    
    # Normalize by dividing by the sum
    probabilities /= probabilities.sum()
    
    return probabilities

def calculate_transmission_delay(data_size, distance, bandwidth):
    """Calculate transmission delay based on Shannon capacity"""
    # (This is a simplified, unused model but we leave it)
    snr = 100 / (1 + distance)  # Simple SNR model
    capacity = bandwidth * np.log2(1 + snr)  # Shannon capacity
    delay = data_size * 8 / capacity  # Convert MB to bits
    return delay

def calculate_processing_delay(computation_demand, computing_resources):
    """Calculate processing delay"""
    return computation_demand / computing_resources
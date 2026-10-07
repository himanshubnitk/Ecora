import numpy as np
from typing import List, Dict, Tuple, Set
from core.entities import Vehicle, AccessPoint, MissionVehicle, CollaborativeVehicle

class VehicularNetwork:
    """Manages network topology and connectivity"""
    
    def __init__(self, area_size: Tuple[float, float], channel_config: dict):
        self.area_size = area_size
        self.mission_vehicles: List[MissionVehicle] = []
        self.collaborative_vehicles: List[CollaborativeVehicle] = []
        self.access_points: List[AccessPoint] = []
        self.connectivity_matrix: Dict[Tuple[int, int], float] = {}
        
        # Load channel model parameters
        self.config = channel_config
        self.B_mhz = 1 # Subchannel bandwidth (1 MHz from config)
        self.noise_power_dbm = self.config['noise_power_dbm']
        self.ple = self.config['path_loss_exponent']
        self.sigma_db = self.config['shadow_fading_std_dev_db']
        self.Pt_ap_dbm = self.config['ap_tx_power_dbm']
        self.Pt_vehicle_dbm = self.config['vehicle_tx_power_dbm']
        
        # Calculate noise power in linear scale (Watts) for 1 MHz bandwidth
        noise_dbm_per_hz = self.noise_power_dbm
        noise_dbm_1mhz = noise_dbm_per_hz + 10 * np.log10(1e6)
        self.N0_watts = 10**((noise_dbm_1mhz - 30) / 10)

    def add_mission_vehicle(self, vehicle: MissionVehicle):
        self.mission_vehicles.append(vehicle)
        
    def add_collaborative_vehicle(self, vehicle: CollaborativeVehicle):
        self.collaborative_vehicles.append(vehicle)
        
    def add_access_point(self, ap: AccessPoint):
        self.access_points.append(ap)
        
    def update_connectivity(self):
        """Update connectivity matrix based on current positions"""
        self.connectivity_matrix.clear()
        
        # V2V connectivity
        for mv in self.mission_vehicles:
            for cv in self.collaborative_vehicles:
                distance = mv.distance_to(cv)
                if distance <= min(mv.communication_range, cv.communication_range):
                    capacity = self.get_channel_capacity_mbps(distance, self.Pt_vehicle_dbm)
                    self.connectivity_matrix[(mv.id, cv.id)] = capacity
                    
        # V2I connectivity
        for mv in self.mission_vehicles:
            for ap in self.access_points:
                distance = ap.distance_to(mv)
                if distance <= min(mv.communication_range, ap.communication_range):
                    capacity = self.get_channel_capacity_mbps(distance, self.Pt_ap_dbm)
                    self.connectivity_matrix[(mv.id, ap.id)] = capacity

    def get_channel_capacity_mbps(self, distance_m: float, tx_power_dbm: float) -> float:
        """
        Calculates channel capacity in Mbps using Shannon-Hartley.
        C = B * log2(1 + SNR)
        """
        if distance_m < 1:
            distance_m = 1  # Avoid log(0)
            
        # 1. Calculate Received Power (Pr) in dBm
        tx_power_watts = 10**((tx_power_dbm - 30) / 10)
        
        # Path Loss (Log-distance model) in dB
        # PL(dB) = PL0 (at 1m) + 10 * n * log10(d/d0) + X_sigma
        # Simplified: Using Friis path loss for d0=1m, f=2.4GHz
        # Let's use a standard urban macro model: PL(dB) = 128.1 + 3.76 * log10(d_km)
        
        # Simpler model based on exponent:
        # Pr(d) = Pt * Gt * Gr * (lambda / 4pi*d)^n
        # Or in dB: Pr(dBm) = Pt(dBm) - PL(dB)
        
        # Path Loss calculation (in dB)
        # Using 3GPP Urban Micro (UMi) path loss model as a proxy
        # f_ghz = 2.4 # Assume 2.4 GHz
        # PL = 30.6 + 26 * np.log10(distance_m) #
        
        # Let's use the PLE from the config
        # PL(dB) = PL_ref (at 1m) + 10 * PLE * log10(d / 1m)
        PL_ref_db = 40.0 # Path loss at 1 meter,
        path_loss_db = PL_ref_db + 10 * self.ple * np.log10(distance_m)
        
        # Add shadow fading (log-normal)
        shadow_fading_db = np.random.normal(0, self.sigma_db)
        
        # Total loss
        total_loss_db = path_loss_db + shadow_fading_db
        
        # Received power in dBm
        received_power_dbm = tx_power_dbm - total_loss_db
        
        # 2. Calculate SNR
        Pr_watts = 10**((received_power_dbm - 30) / 10)
        snr = Pr_watts / self.N0_watts
        
        if snr < 0:
            snr = 0
        
        # 3. Calculate Capacity (Shannon-Hartley)
        # C = B * log2(1 + SNR)
        B_hz = self.B_mhz * 1e6
        capacity_bps = B_hz * np.log2(1 + snr)
        
        capacity_mbps = capacity_bps / 1e6
        
        return max(0.1, capacity_mbps) # Return 0.1 Mbps minimum if > 0


    def get_neighbors(self, vehicle_id: int) -> List[int]:
        # (This function is fine)
        neighbors = []
        for (v1, v2), quality in self.connectivity_matrix.items():
            if v1 == vehicle_id:
                neighbors.append(v2)
            elif v2 == vehicle_id:
                neighbors.append(v1)
        return neighbors
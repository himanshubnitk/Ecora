import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from core.entities import CollaborativeVehicle, AccessPoint
from models.social import SocialNetworkModel

@dataclass
class ServiceCacheRequest:
    cv_id: int
    ap_id: int
    service_id: int
    preference_score: float

class StableMatchingAlgorithm:
    """
    Implements Algorithm 1: Stable Matching Algorithm Based on Mobile Social Contact.
    """
    
    def __init__(self, network, service_popularities, num_services):
        self.network = network
        self.service_popularities = service_popularities # (Miss 6)
        self.num_services = num_services
        self.matching_result: Dict[int, int] = {} 
        
        # (Miss 1) Initialize Social Network Model
        cv_ids = [cv.id for cv in network.collaborative_vehicles]
        ap_ids = [ap.id for ap in network.access_points]
        all_entity_ids = cv_ids + ap_ids
        self.social_model = SocialNetworkModel(all_entity_ids)
        
    def calculate_ap_preference(self, ap: AccessPoint, cv: CollaborativeVehicle) -> float:
        """
        Calculates AP's preference for a CV (Eq. 16)
        YAP_k,n = (θk,n * QoSk,n) / tup_k,n
        """
        
        # 1. Social Connection (θk,n) - (Eq. 9)
        social_connection = self.social_model.get_social_connection(ap.id, cv.id)
        
        # 2. QoS (QoSk,n) - Simplified to channel capacity
        distance = ap.distance_to(cv)
        link_range = min(ap.communication_range, cv.communication_range)
        if distance > link_range:
            return 0.0  # Out of range
            
        tx_power_dbm = self.network.config['ap_tx_power_dbm']
        qos = self.network.get_channel_capacity_mbps(distance, tx_power_dbm)
        if qos <= 0: return 0.0

        # 3. Transmission Time (tup_k,n) (simplified for preference)
        # We assume a reference data size (e.g., 1 Mbit) for preference
        transmission_time = 1.0 / (qos + 0.01) 
        
        # Eq. 16
        preference = (social_connection * qos) / (transmission_time + 0.01)
        return preference
    
    def calculate_cv_preference(self, cv: CollaborativeVehicle, ap: AccessPoint) -> float:
        """
        Calculates CV's preference for an AP (Eq. 17)
        YCV_k,n = (Dk,n * QRk,n) / (1 + e^(-ρ(f)max))
        """
        
        # 1. Movement Correlation (Dk,n) - (Eq. 10)
        distance = ap.distance_to(cv)
        link_range = min(ap.communication_range, cv.communication_range)
        if distance > link_range:
            return 0.0 # Out of range
            
        direction_vec = ap.position - cv.position
        dot_product = np.dot(cv.velocity, direction_vec)
        mu = 1.0 if dot_product > 0 else 0.5 # (Eq. 11)
        
        movement_correlation = 1 - np.exp(-(mu * ap.communication_range) / (distance + 0.01))
        
        # 2. Link Quality (QRk,n) - Simplified to channel capacity
        tx_power_dbm = self.network.config['vehicle_tx_power_dbm']
        link_quality = self.network.get_channel_capacity_mbps(distance, tx_power_dbm)
        if link_quality <= 0: return 0.0
        
        # 3. Service Popularity (ρ(f)max) - (Miss 6)
        # "selects the cache service with the highest popularity"
        popularity = np.max(self.service_popularities)
        
        # Eq. 17
        preference = (movement_correlation * link_quality) / (1 + np.exp(-popularity))
        return preference
    
    def run(self) -> Dict[int, int]:
        """
        Executes Algorithm 1: Gale-Shapley Stable Matching
        """
        cvs = self.network.collaborative_vehicles
        aps = self.network.access_points
        
        if not cvs or not aps:
            return {}
            
        # --- Build Preference Lists ---
        ap_preferences = {} # {ap_id: [cv_id_1, cv_id_2, ...]}
        cv_preferences = {} # {cv_id: [ap_id_1, ap_id_2, ...]}
        
        for ap in aps:
            scores = []
            for cv in cvs:
                score = self.calculate_ap_preference(ap, cv)
                if score > 0:
                    scores.append((cv.id, score))
            scores.sort(key=lambda x: x[1], reverse=True)
            ap_preferences[ap.id] = [cv_id for cv_id, _ in scores]
            
        for cv in cvs:
            scores = []
            for ap in aps:
                score = self.calculate_cv_preference(cv, ap)
                if score > 0:
                    scores.append((ap.id, score))
            scores.sort(key=lambda x: x[1], reverse=True)
            cv_preferences[cv.id] = [ap_id for ap_id, _ in scores]
            
        # --- Run Gale-Shapley (CVs propose to APs) ---
        unmatched_cvs = set(cv.id for cv in cvs)
        cv_next_proposal = {cv.id: 0 for cv in cvs}
        ap_current_matches = {ap.id: [] for ap in aps} # List of matched CVs
        ap_max_conns = {ap.id: ap.max_connections for ap in aps}
        self.matching_result.clear()
        
        # Pre-calculate AP's ranking of CVs for quick lookup
        ap_rankings = {}
        for ap_id, pref_list in ap_preferences.items():
            ap_rankings[ap_id] = {cv_id: rank for rank, cv_id in enumerate(pref_list)}

        while unmatched_cvs:
            try:
                cv_id = unmatched_cvs.pop()
            except KeyError:
                break # All CVs are matched

            if cv_id not in cv_preferences or cv_next_proposal[cv_id] >= len(cv_preferences[cv_id]):
                continue # No more proposals for this CV

            ap_id = cv_preferences[cv_id][cv_next_proposal[cv_id]]
            cv_next_proposal[cv_id] += 1

            if cv_id not in ap_rankings.get(ap_id, {}):
                unmatched_cvs.add(cv_id)
                continue

            if len(ap_current_matches[ap_id]) < ap_max_conns[ap_id]:
                # AP has a free slot
                ap_current_matches[ap.id].append(cv_id)
                self.matching_result[cv_id] = ap_id
            else:
                # AP is full, check for replacement
                worst_cv_id = -1
                worst_rank = -1
                ap_cv_ranks = ap_rankings.get(ap_id, {}) 
                if not ap_cv_ranks:
                    unmatched_cvs.add(cv_id)
                    continue

                for matched_cv_id in ap_current_matches[ap.id]:
                    rank = ap_cv_ranks.get(matched_cv_id, 9999)
                    if rank > worst_rank:
                        worst_rank = rank
                        worst_cv_id = matched_cv_id
                        
                current_cv_rank = ap_cv_ranks.get(cv_id, 9999)
                
                if current_cv_rank < worst_rank:
                    # New CV is better
                    ap_current_matches[ap.id].remove(worst_cv_id)
                    ap_current_matches[ap.id].append(cv_id)
                    self.matching_result[cv_id] = ap_id
                    
                    if worst_cv_id in self.matching_result:
                        del self.matching_result[worst_cv_id]
                    unmatched_cvs.add(worst_cv_id)
                else:
                    # CV is rejected
                    unmatched_cvs.add(cv_id)

        return self.matching_result
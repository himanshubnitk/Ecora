import numpy as np
import networkx as nx
from typing import Dict, Tuple, List

class SocialNetworkModel:
    """
    Implements the Mobile Social Layer from the ECORA paper.
    Computes Social Connection (Eq. 9) from:
    - Interest Similarity (Eq. 6)
    - Social Trust (Eq. 7, 8)
    """
    
    def __init__(self, entity_ids: List[int], num_interests: int = 10):
        self.entity_ids = entity_ids
        self.num_entities = len(entity_ids)
        self.social_graph = nx.Graph()
        self.interest_vectors: Dict[int, np.ndarray] = {}
        self.trust_scores: Dict[Tuple[int, int], float] = {}
        
        print("    > Initializing Mobile Social Layer...")
        self._initialize_social_network(num_interests)
        self._calculate_all_social_trust()
        
    def _initialize_social_network(self, num_interests: int):
        """Initializes social graph with nodes and random interest vectors."""
        for entity_id in self.entity_ids:
            self.social_graph.add_node(entity_id)
            # Each entity has a random "interest" vector
            self.interest_vectors[entity_id] = np.random.rand(num_interests)
            
        # Add random social connections (edges)
        for i in range(self.num_entities):
            v1 = self.entity_ids[i]
            num_connections = np.random.poisson(3) # Avg 3 connections
            for _ in range(num_connections):
                v2 = self.entity_ids[np.random.randint(self.num_entities)]
                if v1 != v2:
                    self.social_graph.add_edge(v1, v2)
                    
    def _calculate_all_social_trust(self):
        """
        Pre-computes social trust (Eq. 7, 8) for all node pairs.
        Uses betweenness centrality as the trust metric.
        """
        print("    > Calculating Social Trust (Betweenness Centrality)...")
        # Eq 7: Betweenness centrality (g_m,y(e) / G_m,y)
        # We use shortest_path_length as a simpler, standard proxy for trust.
        # Trust = 1 / (1 + path_length)
        
        # Eq 8: Normalization
        # The paper's (K+N+M)^2 normalization is non-standard.
        # We will use the standard trust = 1 / (1 + path_length)
        
        # This calculates shortest path length between all node pairs
        path_lengths = dict(nx.all_pairs_shortest_path_length(self.social_graph))
        
        for v1 in self.entity_ids:
            for v2 in self.entity_ids:
                if v1 == v2:
                    self.trust_scores[(v1, v2)] = 1.0
                    continue
                
                if v2 in path_lengths[v1]:
                    length = path_lengths[v1][v2]
                    self.trust_scores[(v1, v2)] = 1.0 / (1.0 + length)
                else:
                    self.trust_scores[(v1, v2)] = 0.0

    def get_interest_similarity(self, v1: int, v2: int) -> float:
        """
        Calculates Cosine Similarity (Eq. 6)
        """
        vec1 = self.interest_vectors.get(v1, np.zeros(10))
        vec2 = self.interest_vectors.get(v2, np.zeros(10))
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 * norm2 > 0:
            return dot_product / (norm1 * norm2)
        return 0.0
    
    def get_social_trust(self, v1: int, v2: int) -> float:
        """Looks up the pre-computed social trust."""
        return self.trust_scores.get((v1, v2), 0.0)

    def get_social_connection(self, v1: int, v2: int) -> float:
        """
        Calculates Overall Social Connection (Eq. 9)
        θm,y = α2*Sm,y + β2*Bm,y
        """
        alpha2, beta2 = 0.5, 0.5 # From paper
        
        similarity = self.get_interest_similarity(v1, v2)
        trust = self.get_social_trust(v1, v2)
        
        return (alpha2 * similarity) + (beta2 * trust)
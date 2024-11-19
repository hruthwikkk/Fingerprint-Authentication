import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

class FingerprintMatcher:
    def __init__(self, threshold=50):
        self.threshold = threshold
        self.templates = {}
    
    def _compute_minutiae_similarity(self, m1, m2):
        """
        Compute similarity between two minutiae points.
        """
        # Extract coordinates and orientations
        x1, y1, type1, orient1 = m1
        x2, y2, type2, orient2 = m2
        
        # Spatial distance
        spatial_dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        
        # Orientation difference (considering circular nature)
        orient_diff = min(abs(orient1 - orient2), 2*np.pi - abs(orient1 - orient2))
        
        # Type difference (0 if same type, 1 if different)
        type_diff = abs(type1 - type2)
        
        # Combined similarity score
        similarity = np.exp(-spatial_dist/50.0) * np.exp(-orient_diff/np.pi) * (1 - 0.5*type_diff)
        
        return similarity
    
    def _match_templates(self, query_features, template_features):
        """
        Match two fingerprint templates using minutiae matching.
        """
        if len(query_features) == 0 or len(template_features) == 0:
            return float('inf')
        
        # Reshape features into minutiae format
        query_minutiae = query_features.reshape(-1, 4)
        template_minutiae = template_features.reshape(-1, 4)
        
        # Compute similarity matrix
        n_query = len(query_minutiae)
        n_template = len(template_minutiae)
        similarity_matrix = np.zeros((n_query, n_template))
        
        for i in range(n_query):
            for j in range(n_template):
                similarity_matrix[i,j] = self._compute_minutiae_similarity(
                    query_minutiae[i], template_minutiae[j])
        
        # Use Hungarian algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
        
        # Compute final matching score
        score = -similarity_matrix[row_ind, col_ind].mean()
        
        return score
    
    def enroll(self, person_id, features):
        """
        Enroll a person's fingerprint features.
        """
        if person_id not in self.templates:
            self.templates[person_id] = []
        self.templates[person_id].append(features)
    
    def match(self, query_features):
        """
        Match query fingerprint against enrolled templates.
        Returns (best_match_id, score)
        """
        best_score = float('inf')
        best_match = None
        
        for person_id, template_list in self.templates.items():
            for template in template_list:
                score = self._match_templates(query_features, template)
                
                if score < best_score:
                    best_score = score
                    best_match = person_id
        
        return best_match, best_score
    
    def verify(self, query_features, claimed_id):
        """
        Verify if query fingerprint matches claimed identity.
        """
        if claimed_id not in self.templates:
            return False, float('inf')
        
        best_score = float('inf')
        for template in self.templates[claimed_id]:
            score = self._match_templates(query_features, template)
            best_score = min(best_score, score)
        
        return best_score <= self.threshold, best_score
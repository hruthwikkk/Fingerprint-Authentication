import cv2
import numpy as np
from scipy import ndimage

class MinutiaeExtractor:
    def __init__(self):
        self.crossing_number_lookup = {
            1: 'ridge_ending',
            3: 'bifurcation'
        }
        self.min_distance = 10  # Minimum distance between minutiae points
    
    def compute_crossing_number(self, values):
        """
        Compute crossing number for minutiae detection.
        """
        crossings = 0
        for i in range(8):
            crossings += abs(int(values[i]) - int(values[(i + 1) % 8]))
        return crossings // 2
    
    def is_valid_minutiae(self, x, y, minutiae_list, image_shape):
        """
        Check if minutiae point is valid based on position and distance from others.
        """
        # Check border distance
        border_distance = 20
        if (x < border_distance or x > image_shape[1] - border_distance or
            y < border_distance or y > image_shape[0] - border_distance):
            return False
        
        # Check distance from existing minutiae
        for m in minutiae_list:
            dist = np.sqrt((x - m['x'])**2 + (y - m['y'])**2)
            if dist < self.min_distance:
                return False
        
        return True
    
    def extract_minutiae(self, skeleton):
        """
        Extract minutiae points from skeletonized fingerprint image.
        """
        minutiae = []
        rows, cols = skeleton.shape
        
        # Get ridge orientation
        orientation = np.zeros((rows, cols))
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if skeleton[i, j] == 255:
                    block = skeleton[i-1:i+2, j-1:j+2]
                    orientation[i, j] = np.arctan2(np.sum(block[2,:] - block[0,:]),
                                                 np.sum(block[:,2] - block[:,0]))
        
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if skeleton[i, j] == 255:  # Ridge pixel
                    # Get 8-neighborhood values
                    values = [
                        skeleton[i-1, j] > 0,   # top
                        skeleton[i-1, j+1] > 0, # top-right
                        skeleton[i, j+1] > 0,   # right
                        skeleton[i+1, j+1] > 0, # bottom-right
                        skeleton[i+1, j] > 0,   # bottom
                        skeleton[i+1, j-1] > 0, # bottom-left
                        skeleton[i, j-1] > 0,   # left
                        skeleton[i-1, j-1] > 0  # top-left
                    ]
                    
                    cn = self.compute_crossing_number(values)
                    
                    if cn in self.crossing_number_lookup:
                        if self.is_valid_minutiae(j, i, minutiae, skeleton.shape):
                            minutiae.append({
                                'x': j,
                                'y': i,
                                'type': self.crossing_number_lookup[cn],
                                'orientation': orientation[i, j]
                            })
        
        return minutiae
    
    def extract_features(self, preprocessed_image):
        """
        Extract features from preprocessed fingerprint image.
        Returns feature vector containing minutiae coordinates, types, and orientations.
        """
        minutiae = self.extract_minutiae(preprocessed_image)
        
        # Create feature vector with enhanced information
        features = []
        for m in minutiae:
            features.extend([
                m['x'],
                m['y'],
                1 if m['type'] == 'ridge_ending' else 0,
                m['orientation']
            ])
        
        return np.array(features)
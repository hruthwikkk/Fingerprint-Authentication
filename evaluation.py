import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns

class SystemEvaluator:
    def __init__(self):
        self.genuine_scores = []
        self.impostor_scores = []
    
    def add_comparison(self, score, is_genuine):
        """
        Add a comparison score to the evaluation.
        """
        if is_genuine:
            self.genuine_scores.append(score)
        else:
            self.impostor_scores.append(score)
    
    def plot_score_distributions(self):
        """
        Plot genuine and impostor score distributions.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(self.genuine_scores, bins=50, alpha=0.5, label='Genuine', density=True)
        plt.hist(self.impostor_scores, bins=50, alpha=0.5, label='Impostor', density=True)
        plt.xlabel('Matching Score')
        plt.ylabel('Density')
        plt.title('Score Distributions')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_roc_curve(self):
        """
        Plot ROC curve and calculate AUC.
        """
        # Prepare labels and scores
        y_true = ([1] * len(self.genuine_scores) + 
                 [0] * len(self.impostor_scores))
        scores = self.genuine_scores + self.impostor_scores
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
        
        return roc_auc
    
    def calculate_eer(self):
        """
        Calculate Equal Error Rate (EER).
        """
        y_true = ([1] * len(self.genuine_scores) + 
                 [0] * len(self.impostor_scores))
        scores = self.genuine_scores + self.impostor_scores
        
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        fnr = 1 - tpr
        
        # Find the threshold where FPR and FNR are closest
        eer = fpr[np.nanargmin(np.absolute(fpr - fnr))]
        
        return eer
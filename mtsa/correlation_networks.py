import numpy as np
import abc

class CorrelationNetwork(abc.ABC):
    """
    Base class for implementing correlation between mfccs
    """

    def get_correlation(self, X: np.ndarray):
        """Implement correlation"""

class PearsonCorrelationNetwork(CorrelationNetwork):

    def get_correlation(self, X : np.ndarray):
        C = np.corrcoef(X, rowvar=True)
        return C

class DynamicCorrelationNetwork(CorrelationNetwork):

    def get_correlation(self, X : np.ndarray):
        """
        Calculate dynamic correlation <(i - <i>)(j - <j>)>
        where <x> is the first moment (e.g, expected value)
        """
        X = X.T
        x_min = np.min(X, axis=0)
        x_max = np.max(X, axis=0)
        X = (X - x_min) / (x_max - x_min)
        X = 2*X - 1
        m = np.mean(X, axis=0)
        D = X - m
        return 1/len(D) * D.T.dot(D)
    
import scipy.stats as st
class SpearmanCorrelationNetwork(CorrelationNetwork):

    def get_correlation(self, X : np.ndarray):
        rho, pval = st.spearmanr(X)
        return rho
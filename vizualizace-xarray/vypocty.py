import numpy as np
import warnings
warnings.filterwarnings('ignore')

class MeanCalculator:
    """
    Co to je: Třída sdružující statické metody pro výpočet různých typů průměrů.
    S čím pracuje: S numerickými poli (numpy.ndarray).
    Kde bere data: Data (q) jsou předávána jako argumenty při volání metod.
    """
    
    @staticmethod
    def arithmetic(q: np.ndarray, axis=None) -> float | np.ndarray:
        # aritmetický průměr : A_1(q) = sum_i (q_i) / N
        return np.mean(q, axis=axis)

    @staticmethod
    def geometric(q: np.ndarray, axis=None) -> float | np.ndarray:
        # geometrický průměr : A_log(q) = exp( (sum_i log(q_i)) / N )  = (prod_i q_i)**(1/N)
        safe_q = np.clip(q, 1e-10, None)
        return np.exp(np.mean(np.log(safe_q), axis=axis))

    @staticmethod
    def harmonic(q: np.ndarray, axis=None) -> float | np.ndarray:
        # Harmonický průměr: A_inv(q) = (( sum_i q_i**(-1)) /N)**(-1)
        safe_q = np.clip(q, 1e-10, None)
        return 1.0 / np.mean(1.0 / safe_q, axis=axis)

def get_mean_func(mean_type: str):
    """
    Co to je: Pomocná funkce (Factory pattern) pro výběr správné matematické funkce.
    Typ výstupu: Vrací odkaz na konkrétní funkci (callable).
    S čím pracuje: S textovým řetězcem (název průměru z UI).
    Kde bere data: Textový řetězec je předán z dropdown menu v Marimo rozhraní.
    """
    if mean_type == "Geometrický": return MeanCalculator.geometric
    if mean_type == "Harmonický": return MeanCalculator.harmonic
    return MeanCalculator.arithmetic
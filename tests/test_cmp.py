"""Tests for CMP (Conway-Maxwell-Poisson) distribution."""
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammaln  # pylint: disable=no-name-in-module


def compute_normalizing_constant(lambda_: float,
                                 nu: float,
                                 k_max: int = 100) -> float:
    """Compute the normalizing constant Z(lambda, nu)."""
    ks = np.arange(0, k_max + 1)
    log_terms = ks * np.log(lambda_) - nu * gammaln(ks + 1)
    Z = np.sum(np.exp(log_terms))
    return Z


def cmp_pmf(k: int,
            lambda_: float,
            nu: float,
            Z: float | None = None,
            k_max: int = 100) -> float:
    """Calculate the CMP probability mass function for a given k."""
    if Z is None:
        Z = compute_normalizing_constant(lambda_, nu, k_max)
    log_p = k * np.log(lambda_) - nu * gammaln(k + 1)
    return np.exp(log_p) / Z


def plot_cmp_distribution(lambda_: float, nu: float, k_max: int = 20) -> None:
    """Plot cmp distribution."""
    ks = np.arange(0, k_max + 1)
    Z = compute_normalizing_constant(lambda_, nu, k_max=100)
    ps = [cmp_pmf(k, lambda_, nu, Z) for k in ks]

    plt.figure(figsize=(10, 6))
    plt.bar(ks, ps, color='skyblue', edgecolor='black')
    plt.title(
        f"Conway-Maxwell-Poisson Distribution (lambda={lambda_}, nu={nu})")
    plt.xlabel("k")
    plt.ylabel("P(X = k)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


# Example usage
plot_cmp_distribution(lambda_=100.0, nu=2.8)  # to have mode at 4
# plot_cmp_distribution(lambda_=80.0, nu=3.0)  # to have mode at 4
# plot_cmp_distribution(lambda_=55.0, nu=3.0)  # to have mode at 3
# plot_cmp_distribution(lambda_=10.0, nu=1.0)  # Equivalent to Poisson
# plot_cmp_distribution(lambda_=4.0, nu=0.5)  # Overdispersed
# plot_cmp_distribution(lambda_=4.0, nu=2.0)  # Underdispersed

"""
Bayesian analysis functions for industry-specific datasets.

This module provides helper functions to perform Bayesian analyses on
three open datasets commonly used in teaching Bayesian statistics.  The
datasets correspond to problems in the telecommunications, banking and
health‑insurance sectors:

* **Telecommunications** – The ``churn.csv`` file contains
  demographic information, service subscriptions and churn labels for
  customers of a fictitious telecom operator.  Understanding what
  drives churn is crucial for telcos dealing with increased
  competition and shifting customer expectations.

* **Banking** – The ``credit_card.csv`` file records
  customer demographics, credit limits and payment histories along
  with a binary indicator of whether the cardholder defaulted on the
  loan.  Rising interest rates and changing underwriting standards
  have made it important for banks to accurately estimate default
  rates across segments.

* **Health Insurance** – The ``insurance.csv`` dataset contains
  individual characteristics such as age, body‑mass index (BMI),
  number of children and smoking status together with the cost of
  medical insurance.  Health‑insurance costs are increasing and are
  strongly influenced by many variables.
  A Bayesian regression model can capture the uncertainty in how
  different factors contribute to charges.

The functions defined in this module can be used as building blocks
for a larger analysis script or notebook.  They assume that the
datasets have been downloaded locally (for example into a ``data/``
directory) because direct HTTP requests from Python are blocked in
this environment.  See the accompanying report for instructions on
downloading the CSV files via your web browser.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple
from scipy import stats

@dataclass
class BetaPosterior:
    """Represents the posterior Beta distribution after observing
    binomial data.

    Attributes
    ----------
    alpha : float
        Shape parameter α of the Beta posterior.
    beta : float
        Shape parameter β of the Beta posterior.
    """
    alpha: float
    beta: float

    def mean(self) -> float:
        """Posterior mean of the probability."""
        return self.alpha / (self.alpha + self.beta)

    def credible_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """Returns the central credible interval for the Beta posterior.

        Parameters
        ----------
        level : float, optional
            Probability mass of the credible interval (default 0.95 for a
            95% interval).

        Returns
        -------
        (float, float)
            Lower and upper bounds of the central credible interval.
        """
        lower = (1.0 - level) / 2
        upper = 1.0 - lower
        return (
            stats.beta.ppf(lower, self.alpha, self.beta),
            stats.beta.ppf(upper, self.alpha, self.beta),
        )


def compute_beta_posterior(successes: int, trials: int,
                           alpha_prior: float = 1.0,
                           beta_prior: float = 1.0) -> BetaPosterior:
    """Compute the posterior Beta distribution for a binomial proportion.

    A Beta prior with parameters ``alpha_prior`` and ``beta_prior`` is
    updated using binomial data with ``successes`` successes out of
    ``trials`` trials.  The resulting posterior is ``Beta(α + successes,
    β + failures)`` where ``failures = trials - successes``.

    Parameters
    ----------
    successes : int
        Number of positive outcomes (e.g. churned customers or defaults).
    trials : int
        Total number of trials (e.g. number of customers in the group).
    alpha_prior : float, optional
        α parameter of the Beta prior (default 1.0, a uniform prior).
    beta_prior : float, optional
        β parameter of the Beta prior (default 1.0, a uniform prior).

    Returns
    -------
    BetaPosterior
        An object representing the Beta posterior.
    """
    if trials < 0:
        raise ValueError("trials must be non‑negative")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between 0 and trials")
    alpha_post = alpha_prior + successes
    beta_post = beta_prior + (trials - successes)
    return BetaPosterior(alpha_post, beta_post)


def churn_posterior_by_contract(telco: pd.DataFrame,
                                alpha_prior: float = 1.0,
                                beta_prior: float = 1.0
                                ) -> Dict[str, BetaPosterior]:
    """Compute churn rate posteriors for each contract type.

    The function expects a telco churn DataFrame containing a ``Contract``
    column (categorical with values like ``'Month-to-month'``, ``'One year'``
    and ``'Two year'``) and a ``Churn`` column indicating whether the
    customer churned (values ``'Yes'`` or ``'No'``).  For each contract
    category, a Beta posterior of the churn probability is computed using
    a uniform Beta(1,1) prior by default.

    Parameters
    ----------
    telco : pd.DataFrame
        DataFrame of telco customers.
    alpha_prior : float, optional
        α parameter of the Beta prior.
    beta_prior : float, optional
        β parameter of the Beta prior.

    Returns
    -------
    Dict[str, BetaPosterior]
        A mapping from contract type to its posterior distribution.
    """
    # Normalize churn column to boolean 1/0
    churn_binary = telco['Churn'].astype(str).str.strip().str.lower().map({
        'yes': 1, 'no': 0
    })
    # Ensure the mapping does not produce NaNs
    if churn_binary.isnull().any():
        raise ValueError("Unexpected values in 'Churn' column.")

    results: Dict[str, BetaPosterior] = {}
    for contract, group in telco.groupby('Contract'):
        successes = churn_binary.loc[group.index].sum()
        n = len(group)
        results[contract] = compute_beta_posterior(
            int(successes), int(n), alpha_prior, beta_prior
        )
    return results


def default_posterior_by_credit_quartile(credit: pd.DataFrame,
                                         alpha_prior: float = 1.0,
                                         beta_prior: float = 1.0
                                         ) -> Dict[str, BetaPosterior]:
    """Compute default rate posteriors by credit‑limit quartile.

    The banking dataset contains a ``LIMIT_BAL`` column (credit limit) and a
    ``DEFAULT`` column (1 for default, 0 for non‑default).  This
    function divides customers into quartiles based on ``LIMIT_BAL`` and
    computes a Beta posterior for the default probability in each quartile.

    Parameters
    ----------
    credit : pd.DataFrame
        DataFrame of credit card clients.
    alpha_prior : float, optional
        α parameter of the Beta prior.
    beta_prior : float, optional
        β parameter of the Beta prior.

    Returns
    -------
    Dict[str, BetaPosterior]
        A mapping from quartile label to posterior distribution.
    """
    # Validate required columns
    if 'LIMIT_BAL' not in credit.columns or 'DEFAULT' not in credit.columns:
        raise KeyError("Expected columns 'LIMIT_BAL' and 'DEFAULT' in credit dataset.")
    # Create quartile labels
    quartiles = pd.qcut(credit['LIMIT_BAL'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    credit = credit.assign(limit_quartile=quartiles)
    results: Dict[str, BetaPosterior] = {}
    for quartile, group in credit.groupby('limit_quartile'):
        successes = int(group['DEFAULT'].sum())
        n = len(group)
        results[str(quartile)] = compute_beta_posterior(
            successes, n, alpha_prior, beta_prior
        )
    return results


def prepare_insurance_data(insurance: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
    """Prepare the insurance dataset for Bayesian linear regression.

    This helper function encodes categorical variables into dummy
    indicators, standardizes continuous predictors, and returns a
    design matrix ``X``, a response vector ``y`` and the names of
    predictors.  The intercept column is included automatically.

    Parameters
    ----------
    insurance : pd.DataFrame
        Raw insurance dataset with columns ``age``, ``sex``, ``bmi``,
        ``children``, ``smoker``, ``region`` and ``charges``.

    Returns
    -------
    X : np.ndarray
        Design matrix of shape (n, p) including an intercept.
    y : np.ndarray
        Response vector of log‑transformed charges (to stabilize
        variance).
    feature_names : pd.Index
        Names of the predictor columns including the intercept.
    """
    # Copy to avoid modifying original
    df = insurance.copy()
    # Response variable: log of charges (charges are strictly positive)
    y = np.log(df['charges'].values)
    # Process categorical variables
    df['sex'] = df['sex'].str.lower().map({'male': 1, 'female': 0})
    df['smoker'] = df['smoker'].str.lower().map({'yes': 1, 'no': 0})
    # One‑hot encode region
    region_dummies = pd.get_dummies(df['region'], prefix='region', drop_first=True)
    # Construct design matrix with intercept
    X_continuous = df[['age', 'bmi', 'children', 'sex', 'smoker']]
    # Standardize continuous predictors (zero mean, unit variance)
    X_standardized = (X_continuous - X_continuous.mean()) / X_continuous.std()
    X_full = pd.concat([pd.Series(1.0, index=df.index, name='Intercept'),
                        X_standardized,
                        region_dummies], axis=1)
    return X_full.values, y, X_full.columns


def bayesian_linear_regression(X: np.ndarray, y: np.ndarray,
                               beta0: np.ndarray | None = None,
                               V0: np.ndarray | None = None,
                               a0: float = 1.0, b0: float = 1.0
                               ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Perform Bayesian linear regression using a conjugate
    normal–inverse‑gamma prior.

    This function implements the closed‑form update formulas for the
    normal‑inverse‑gamma prior as described in *Think Bayes* and many
    statistics textbooks.  The model is::

        y | β, σ² ~ Normal(X β, σ² I)
        β | σ² ~ Normal(β₀, σ² V₀)
        σ² ~ InvGamma(a₀, b₀)

    The posterior is::

        β | σ², y ~ Normal(βₙ, σ² Vₙ)
        σ² ~ InvGamma(aₙ, bₙ)

    where

        Vₙ = (XᵀX + V₀⁻¹)⁻¹,
        βₙ = Vₙ (Xᵀy + V₀⁻¹ β₀),
        aₙ = a₀ + n/2,
        bₙ = b₀ + 0.5 (yᵀy + β₀ᵀ V₀⁻¹ β₀ - βₙᵀ Vₙ⁻¹ βₙ).

    We return the posterior parameters ``βₙ``, ``Vₙ`` and the scalar
    hyperparameters ``aₙ`` and ``bₙ``.  Samples from the posterior can
    then be drawn by first sampling ``σ²`` from an inverse‑gamma
    distribution and then ``β`` from a normal distribution conditional
    on ``σ²``.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Design matrix with intercept.
    y : np.ndarray, shape (n,)
        Response vector (log‑charges in this analysis).
    beta0 : np.ndarray, shape (p,), optional
        Prior mean vector.  Defaults to a zero vector.
    V0 : np.ndarray, shape (p, p), optional
        Prior covariance scaling matrix.  If not provided, an
        identity matrix is used implying each coefficient has a
        moderately diffuse prior.
    a0 : float, optional
        Shape parameter of the inverse‑gamma prior for σ².
    b0 : float, optional
        Scale parameter of the inverse‑gamma prior for σ².

    Returns
    -------
    β_n : np.ndarray
        Posterior mean vector.
    V_n : np.ndarray
        Posterior covariance scaling matrix.
    a_n : float
        Posterior shape parameter for σ².
    b_n : float
        Posterior scale parameter for σ².
    """
    n, p = X.shape
    # Set default prior mean and covariance if not provided
    if beta0 is None:
        beta0 = np.zeros(p)
    if V0 is None:
        V0 = np.eye(p)
    # Compute precision matrices
    V0_inv = np.linalg.inv(V0)
    XtX = X.T @ X
    XtX_plus_V0_inv = XtX + V0_inv
    Vn = np.linalg.inv(XtX_plus_V0_inv)
    # Posterior mean
    Xt_y = X.T @ y
    beta_n = Vn @ (Xt_y + V0_inv @ beta0)
    # Posterior hyperparameters for σ²
    a_n = a0 + n / 2.0
    # Compute b_n using the standard formula
    # Compute yᵀy, β₀ᵀ V₀⁻¹ β₀, and βₙᵀ Vₙ⁻¹ βₙ
    yTy = float(y.T @ y)
    beta0_term = float(beta0.T @ V0_inv @ beta0)
    beta_n_term = float(beta_n.T @ np.linalg.inv(Vn) @ beta_n)
    b_n = b0 + 0.5 * (yTy + beta0_term - beta_n_term)
    return beta_n, Vn, a_n, b_n


def sample_posterior(beta_n: np.ndarray, Vn: np.ndarray, a_n: float, b_n: float,
                     num_samples: int = 1000,
                     random_state: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Draw samples from the posterior of a normal–inverse‑gamma model.

    This function returns joint samples of regression coefficients and
    the error variance.  It first draws ``σ²`` samples from an
    inverse‑gamma distribution with shape ``a_n`` and scale ``b_n``, and
    then draws ``β`` samples from a multivariate normal with mean
    ``β_n`` and covariance ``σ² V_n`` for each draw.

    Parameters
    ----------
    beta_n : np.ndarray
        Posterior mean vector of coefficients.
    Vn : np.ndarray
        Posterior covariance scaling matrix.
    a_n : float
        Posterior shape parameter for σ².
    b_n : float
        Posterior scale parameter for σ².
    num_samples : int, optional
        Number of joint samples to draw (default 1000).
    random_state : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    beta_samples : np.ndarray, shape (num_samples, p)
        Draws of regression coefficients.
    sigma2_samples : np.ndarray, shape (num_samples,)
        Draws of the error variance.
    """
    rng = np.random.default_rng(random_state)
    p = beta_n.shape[0]
    # Sample sigma^2 from inverse-gamma
    sigma2_samples = stats.invgamma(a=a_n, scale=b_n).rvs(size=num_samples, random_state=rng)
    # Draw beta samples conditional on each sigma^2
    beta_samples = np.empty((num_samples, p))
    # Compute the Cholesky decomposition once for speed
    # We draw standard normal z and scale by sqrt(sigma2) * L where L @ L^T = Vn
    L = np.linalg.cholesky(Vn)
    for i in range(num_samples):
        z = rng.standard_normal(p)
        beta_samples[i] = beta_n + np.sqrt(sigma2_samples[i]) * (L @ z)
    return beta_samples, sigma2_samples

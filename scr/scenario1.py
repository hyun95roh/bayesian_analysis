import yfinance as yf
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# 1. Load Data
df = yf.download('AAPL', start='2020-09-21', end='2025-09-21')
df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
returns = df['LogReturn'].dropna().values  # ~1,260 daily returns

# 2. Bayesian Volatility Model
with pm.Model() as model:
    # Priors: Normal-Inverse-Gamma for mean (mu) and variance (sigma2)
    mu = pm.Normal('mu', mu=0, sigma=0.01)
    sigma = pm.InverseGamma('sigma', alpha=2, beta=0.1)
    # Likelihood
    returns_obs = pm.Normal('returns_obs', mu=mu, sigma=sigma, observed=returns)
    # Predictive variable for future returns
    pred_returns = pm.Normal('pred_returns', mu=mu, sigma=sigma, shape=30)  # 30-day forecast
    # Sample posterior
    trace = pm.sample(1000, tune=1000, return_inferencedata=True)

# 3. Posterior Predictive Sampling
with model:
    pred_trace = pm.sample_posterior_predictive(trace, var_names=['pred_returns'])

# 4. Extract Posterior and Predictive Samples
posterior = az.extract(trace)
mu_samples = posterior['mu'].values  # Shape: (4000,) after flattening chains
sigma_samples = posterior['sigma'].values  # Shape: (4000,)
pred_samples = pred_trace.posterior_predictive['pred_returns'].values  # Shape: (chains, draws, 30)

# Save trace for reproducibility
az.to_netcdf(trace, 'trace.nc')
az.to_netcdf(pred_trace, 'pred_trace.nc')

# 5. Interactive Visualization
fig = make_subplots(rows=2, cols=2, subplot_titles=('Prior vs Posterior: Mu', 'Prior vs Posterior: Sigma',
                                                    'MCMC Trace: Mu', 'Predictive Returns (30 Days)'))

# Prior vs Posterior: Mu
fig.add_trace(go.Histogram(x=np.random.normal(0, 0.01, 1000), name='Prior Mu', opacity=0.5), row=1, col=1)
fig.add_trace(go.Histogram(x=mu_samples, name='Posterior Mu', opacity=0.5), row=1, col=1)

# Prior vs Posterior: Sigma
fig.add_trace(go.Histogram(x=np.random.gamma(2, 0.1, 1000), name='Prior Sigma', opacity=0.5), row=1, col=2)
fig.add_trace(go.Histogram(x=sigma_samples, name='Posterior Sigma', opacity=0.5), row=1, col=2)

# MCMC Trace: Mu
fig.add_trace(go.Scatter(x=np.arange(len(mu_samples)), y=mu_samples, mode='lines', name='Mu Trace'), row=2, col=1)

# Predictive Returns: Mean and 95% CI
pred_mean = pred_samples.mean(axis=(0, 1))  # Mean over chains and draws
pred_ci = np.percentile(pred_samples, [2.5, 97.5], axis=(0, 1))  # 95% credible interval
fig.add_trace(go.Scatter(x=np.arange(30), y=pred_mean, mode='lines', name='Mean Pred Returns'), row=2, col=2)
fig.add_trace(go.Scatter(x=np.arange(30), y=pred_ci[0], mode='lines', name='95% CI Lower', line=dict(dash='dash')), row=2, col=2)
fig.add_trace(go.Scatter(x=np.arange(30), y=pred_ci[1], mode='lines', name='95% CI Upper', line=dict(dash='dash')), row=2, col=2)

fig.update_layout(title='Bayesian Volatility Analysis for AAPL', showlegend=True)
fig.write_html('volatility_viz.html')  # Export for GitHub Pages
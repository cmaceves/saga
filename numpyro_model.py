"""
Use numpyro to model beta distributions of frequencies.
"""
import sys
import copy
import numpyro
import numpy as np
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
import jax.numpy as jnp
import pandas as pd
from jax import random
from numpyro.diagnostics import hpdi
from numpyro.infer import Predictive
import matplotlib.pyplot as plt
from scipy.stats import beta

df = pd.read_csv('file_5_sorted.calmd.tsv', sep='\t')
alt_freq = df['ALT_FREQ'].tolist()
ref_freq = [x/y for x,y in zip(df['REF_DP'].tolist(), df['TOTAL_DP'].tolist())]

total_freq = copy.deepcopy(alt_freq)
total_freq.extend(ref_freq)
#values cannot == 0 or == 1
total_freq = [float(round(x,3)) for x in total_freq if 0 < round(x,3) < 1]

def model(data, n):
    # define weights with a dirichlet prior
    weights = numpyro.sample('weights', dist.Dirichlet(concentration=jnp.ones(n)))
    
    with numpyro.plate('components', n):
        # define alpha and beta  for n Beta distributions
        alpha = numpyro.sample('alpha', dist.Gamma(1)) # Using gamma prior for now. Probably not great. 
        beta = numpyro.sample('beta', dist.Gamma(1))

    # define beta distributions
    with numpyro.plate('data', len(data)):
        assignment = numpyro.sample('assignment', dist.Categorical(weights))
        numpyro.sample('obs', dist.Beta(alpha[assignment], beta[assignment]), obs=data)

data = jnp.array(total_freq)
n_val = 10

# Run mcmc with 2 components 
kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup=5, num_samples=10)
mcmc.run(random.PRNGKey(112358), data=data, n=n_val)
mcmc.print_summary()
samples = mcmc.get_samples()

fig, ax = plt.subplots(1, 1)
cm = plt.get_cmap('gist_rainbow')
ax.set_prop_cycle(color=[cm(1.*i/n_val) for i in range(n_val)])
for n in range(n_val):
    alpha = np.mean(samples['alpha'][:,n])
    betas = np.mean(samples['beta'][:,n])
    weight = np.mean(samples['weights'][:,n])
    x = np.linspace(beta.ppf(0.01, alpha, betas),
                    beta.ppf(0.99, alpha, betas), 100)
    ax.plot(x, beta.pdf(x, alpha, betas),
           '-', lw=3, alpha=0.6, label='beta pdf 0')
    mean, var, skew, kurt = beta.stats(alpha, betas, moments='mvsk')
    print(n, mean, weight)

plt.savefig("./analysis/figures/test.png")
"""
samples = mcmc.get_samples()
# Try doing more checks here .. 
predictive = Predictive(model, samples)
predictions = predictive(random.PRNGKey(112358), data = jnp.array(data), n = 2)["obs"]
print(samples.keys())
"""


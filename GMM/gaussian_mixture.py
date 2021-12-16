import jax.numpy as jnp
import jax
from jax.scipy.special import logsumexp
import numpy as np

import numpyro.distributions as dists
from numpyro import sample, plate


class GaussianMixtureModel(dists.Distribution):

    def __init__(self, mixture_probabilities, mixture_locs, mixture_scales, validate_args = None):
        self._pis = jnp.array(mixture_probabilities)
        self._locs = jnp.array(mixture_locs)
        self._scales = jnp.array(mixture_scales)

        batch_shape = ()
        event_shape = self._locs.shape[1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        component_key, samples_key = jax.random.split(key)
        zs = dists.CategoricalProbs(self._pis).sample(component_key, sample_shape)
        xs = dists.Normal(self._locs[zs], self._scales[zs]).sample(samples_key)
        return xs

    def log_prob(self, value): # [batch_size, 2]
        per_component_log_prob = jax.vmap(
            lambda loc, scale: dists.Normal(loc, scale).log_prob(value),
            out_axes=-1
        )(self._locs, self._scales)

        log_pis = jnp.log(self._pis)

        # sum log-likelihood contributions from event dimensions
        per_component_log_prob = per_component_log_prob.sum(axis=-2)

        # "sum" over components
        loglik = logsumexp(per_component_log_prob + log_pis, axis=-1)
        return loglik

def model(xs, N, k=5, d=2):
    pis = sample('pis', dists.Dirichlet(jnp.ones(k)))

    with plate('component_priors', k, k):
        mus = sample('locs', dists.MultivariateNormal(jnp.zeros((d,)), jnp.eye(d)), sample_shape=(k,))
        taus = sample('taus', dists.InverseGamma(1, 1), sample_shape=(k,))

    batch_size = xs.shape[0]
    with plate('batch', N, batch_size):
        return sample(
            'xs', GaussianMixtureModel(pis, mus, taus), obs=xs, sample_shape=(batch_size,)
        )

from numpyro.infer.autoguide import AutoDiagonalNormal
guide = AutoDiagonalNormal(model)

from numpyro.infer import Trace_ELBO, SVI
from numpyro.optim import Adam, Adagrad
from d3p.svi import DPSVI
from d3p.dputil import approximate_sigma
from d3p.minibatch import subsample_batchify_data
from tqdm import tqdm

num_iter = 1000
q = 0.003

def infer(data, rng_key, dp_scale=1., nonprivate=False):
    batch_rng_key, svi_rng_key = jax.random.split(rng_key, 2)
    # set up minibatch sampling
    batch_size = int(len(data)*q)
    batchifier_init, get_batch = \
        subsample_batchify_data((data,), batch_size)
    _, batchifier_state = batchifier_init(batch_rng_key)

    loss = Trace_ELBO()
    optimiser = Adam(1e-3)
    if nonprivate:
        print("NO DP!!!")
        dpsvi = SVI(model, guide, optimiser, loss, N=len(data))
    else:
        clip_threshold = 1.
        dpsvi = DPSVI(model, guide, optimiser, loss, clip_threshold, dp_scale, N=len(data))

    svi_state = dpsvi.init(svi_rng_key, *get_batch(0, batchifier_state))

    @jax.jit
    def run_iteration_block(start_iter, num_iter, svi_state):
        def inner(i, var):
            svi_state, loss = var
            data_batch, = get_batch(i, batchifier_state)
            svi_state, local_loss = dpsvi.update(svi_state, data_batch)
            return svi_state, loss + local_loss / num_iter
        return jax.lax.fori_loop(start_iter, start_iter + num_iter, inner, (svi_state, 0.))

    # run inference
    iter_per_block = 100
    num_blocks = num_iter // iter_per_block
    losses = np.zeros(num_blocks)
    params = [None] * num_blocks
    for i in range(num_blocks):
        svi_state, avg_loss = run_iteration_block(i*iter_per_block, iter_per_block, svi_state)
        losses[i] = avg_loss
        params[i] = dpsvi.get_params(svi_state)
    return params, losses


def generate_data():
    rng = np.random.RandomState(123)

    n_samples = 2000

    R = 2
    d = 2
    ms = np.array([[0, 0], [R, R],[-R, -R],[-R, R],[R, -R]])
    k = len(ms)
    ps = np.ones(k)/k
    ts = 0.5*np.ones(k)

    zs = np.array([rng.multinomial(1, ps) for _ in range(n_samples)]).T
    xs = [z[:, np.newaxis] * rng.multivariate_normal(m, t*np.eye(2), size=n_samples)
        for z, m, t in zip(zs, ms, ts)]
    data = np.sum(np.dstack(xs), axis=2)

    n_test = 100
    test_zs = np.array([rng.multinomial(1, ps) for _ in range(n_test)]).T
    test_xs = [z[:, np.newaxis] * rng.multivariate_normal(m, t*np.eye(2), size=n_test)
        for z, m, t in zip(test_zs, ms, ts)]
    test_data = np.sum(np.dstack(test_xs), axis=2)

    return data, test_data, ms, ts, ps

import matplotlib.pyplot as plt
def plot_result(params, mus_ori, taus_sq_ori):
    colors = ['red', 'blue', 'yellow', 'orange', 'turquoise']
    pis = guide.median(params)['pis']
    mus = guide.median(params)['locs']
    taus = guide.median(params)['taus']


    _, ax = plt.subplots()
    ax.axis('equal')
    ax.scatter(data[:, 0], data[:, 1], color='green', alpha=0.2, s=2.0)

    for n, color in enumerate(colors):
        v = 2. * np.sqrt(2.) * taus[n]
        ax.add_artist(plt.Circle(mus[n], radius=v, color=color, fill=False, linestyle='dashed'))
    for n in range(len(mus_ori)):
        v = 2. * np.sqrt(2.) * np.sqrt(taus_sq_ori[n])
        ax.add_artist(plt.Circle(mus_ori[n], radius=v, color='black', fill=False))

    ax.set_xlim((-6, 6))
    ax.set_ylim((-6, 6))



def act(sigma, delta_tot, delta_prime, T, q):
	d_iter = (delta_tot-delta_prime)/(T*q)
	sigma_prime = np.log(1+q*(np.exp(np.sqrt(2*np.log(1.25/d_iter))/sigma)-1))
	return np.sqrt(2*T*np.log(1.25/delta_prime))*sigma_prime+T*sigma_prime*(np.exp(sigma_prime)-1)

from scipy.optimize import minimize

def find_sigma_act(epsilon, delta_tot, T, q):
	def fun(sigma):
	    return np.abs(2*act(2*sigma, 0.5*delta_tot, 0.1*delta_tot, T, q)-epsilon)
	tmp = minimize(fun, 1.0, method='Nelder-Mead', tol = 1e-5,  options={'maxiter':100})
	return tmp['x'][0], tmp['fun']

import argparse
from jax.random import PRNGKey
from numpyro.handlers import trace, seed, substitute
import pickle

from matplotlib import rc
import matplotlib.pyplot as plt


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.style.use('ggplot')

rc('text', usetex=True)

SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# plt.rcParams["figure.figsize"] = (3,2.7)

dpi = 300


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=1.)
    parser.add_argument("--no-dp", action='store_true', default=False)
    parser.add_argument("--use_cached", action='store_true', default=True, help="If this flag is set, skip inference and only perform plotting based on pickle'd results.")

    args = parser.parse_args()

    seeds = range(5)
    epsilons = np.geomspace(0.1, 10, 5)
    # epsilons = [1.]
    dp_scales = np.empty(len(epsilons))

    for i, epsilon in enumerate(epsilons):
        dp_scale, _ = find_sigma_act(epsilon, delta_tot=0.001, T=num_iter, q=q)
        print(f"dp_scale for eps={epsilon}: {dp_scale}")
        dp_scales[i] = dp_scale


    if not args.use_cached:
        data, test_data, mus, taus_sq, pis = generate_data()

        results = dict() # results[eps] = log_probs on test

        for epsilon, dp_scale in zip(epsilons, dp_scales):
            log_likelihoods = np.empty(len(seeds))

            for run_no, rng_seed in tqdm(enumerate(seeds)):
                np.random.seed(rng_seed)

                all_params, all_losses = infer(
                    data, PRNGKey(rng_seed), dp_scale, args.no_dp
                )

                params = all_params[-1]

                with trace() as tr:
                    seed(substitute(model, guide.median(params)), PRNGKey(0))(test_data, len(test_data)) # PRNGKey plays no role as all "sample" values are set in this case
                    log_likelihoods[run_no] = tr['xs']['fn'].log_prob(test_data).mean()

            results[epsilon] = {'raw': log_likelihoods, 'avg': np.mean(log_likelihoods), 'std': np.std(log_likelihoods)}

        with open('results.pickle', 'wb') as f:
            pickle.dump(results, f)

    with open('results.pickle', 'rb') as f:
        results = pickle.load(f)

    with open('mog_preds_advi.p', 'rb') as f:
        dpvi_paper_results = pickle.load(f, encoding='latin1')

    result_means = np.array([results[eps]['avg'] for eps in epsilons])
    result_stds = np.array([results[eps]['std'] for eps in epsilons])
    result_ns = np.array([len(results[eps]['raw']) for eps in epsilons])
    result_errs = result_stds / np.sqrt(result_ns)
    plt.errorbar(epsilons, result_means, yerr=result_errs, fmt='-x', label='d3p')
    plt.fill_between(epsilons, result_means - result_errs, result_means + result_errs,
        alpha=.3
    )

    for eps in dpvi_paper_results.keys():
        dpvi_paper_results[eps]['avg'] = np.mean(dpvi_paper_results[eps]['raw'])
        dpvi_paper_results[eps]['std'] = np.std(dpvi_paper_results[eps]['raw'])

    dpvi_paper_result_means = np.array([dpvi_paper_results[eps]['avg'] for eps in epsilons])
    dpvi_paper_result_stds = np.array([dpvi_paper_results[eps]['std'] for eps in epsilons])
    dpvi_paper_result_ns = np.array([len(dpvi_paper_results[eps]['raw']) for eps in epsilons])
    dpvi_paper_result_errs = dpvi_paper_result_stds / np.sqrt(dpvi_paper_result_ns)
    plt.errorbar(epsilons, dpvi_paper_result_means, yerr=dpvi_paper_result_errs, fmt='-o', label='Jälkö et al.')
    plt.fill_between(epsilons,
        dpvi_paper_result_means - dpvi_paper_result_errs, dpvi_paper_result_means + dpvi_paper_result_errs,
        alpha=.3
    )

    plt.title('GMM log-likelihood over $\\varepsilon$')

    plt.ylabel('Log-likelihood')
    plt.xlabel('$\\varepsilon$')
    plt.xscale('log')
    plt.legend(loc='lower right')
    plt.savefig('gmm_logliks.pdf', bbox_inches='tight')
    plt.show()


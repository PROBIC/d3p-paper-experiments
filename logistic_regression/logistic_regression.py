import jax.numpy as jnp
import jax
from jax.nn import sigmoid
from numpyro import sample, param, plate
from numpyro.infer import Trace_ELBO, SVI
from numpyro.optim import Adam
from numpyro.distributions import Normal, BernoulliLogits
from jax.random import PRNGKey

import scipy.stats

from d3p.minibatch import subsample_batchify_data
from d3p.svi import DPSVI
from d3p.dputil import approximate_sigma

from tqdm import tqdm
import argparse

import numpy as np

def model(xs, ys, ls, gs, N):
    batch_size, D = xs.shape
    L, K = gs.shape

    M = sample('M', Normal(0, 4), sample_shape=(D, K))

    with plate('group', L, L):
        etas = gs @ M.T
        ws = sample('ws', Normal(etas, 1.).to_event(1))

    with plate('batch', N, batch_size):
        thetas = jnp.einsum("nd,nd->n", xs, ws[ls])
        ys = sample('ys', BernoulliLogits(thetas), obs=ys)

def guide(xs, ys, ls, gs, N):
    _, D = xs.shape
    _, K = gs.shape

    M_loc = param('M_loc', np.random.randn(D, K))
    M_scale = jnp.exp(param('M_scale_log', np.random.randn(D, K)))
    sample('M', Normal(M_loc, M_scale))

def infer(data, labels, ls, gs, batch_size, num_iter, epsilon, delta, rng_key):
    batch_rng_key, svi_rng_key = jax.random.split(rng_key, 2)
    # set up minibatch sampling
    batchifier_init, get_batch = \
        subsample_batchify_data((data, labels, ls), batch_size)
    _, batchifier_state = batchifier_init(batch_rng_key)

    loss = Trace_ELBO()
    optimiser = Adam(1e-3)
    if epsilon is None:
        dpsvi = SVI(model, guide, optimiser, loss, N=len(data), gs=gs)
    else:
        q = batch_size / len(data)
        dp_scale, _, _ = approximate_sigma(epsilon, delta, q, num_iter, maxeval=40)
        print(f"dp scale is {dp_scale}")
        clip_threshold = 1.
        dpsvi = DPSVI(model, guide, optimiser, loss, clip_threshold, dp_scale, N=len(data), gs=gs)

    svi_state = dpsvi.init(svi_rng_key, *get_batch(0, batchifier_state))

    @jax.jit
    def run_iteration_block(start_iter, num_iter, svi_state):
        def inner(i, var):
            svi_state, loss = var
            data_batch, label_batch, ls_batch = get_batch(i, batchifier_state)
            svi_state, local_loss = dpsvi.update(svi_state, data_batch, label_batch, ls_batch)
            return svi_state, loss + local_loss / num_iter
        return jax.lax.fori_loop(start_iter, start_iter + num_iter, inner, (svi_state, 0.))

    # run inference
    iter_per_block = 10
    num_blocks = num_iter // iter_per_block
    losses = np.zeros(num_blocks)
    params = [None] * num_blocks
    for i in tqdm(range(num_blocks)):
        svi_state, avg_loss = run_iteration_block(i*iter_per_block, iter_per_block, svi_state)
        losses[i] = avg_loss
        params[i] = dpsvi.get_params(svi_state)
    return params, losses


def generate_data(L, D, K, N):
    np.random.seed(0)
    xs = np.random.randn(2, N, D)
    ls = np.random.randint(0, L, (2, N))

    M_ori = np.random.randn(D, K)
    gs = np.eye(L, K)
    ws_ori = np.zeros((L, D))
    ys = np.zeros((2, N))
    thetas_ori = np.zeros((2, N))

    for l in range(L):
        ws_ori[l] = np.matmul(M_ori, gs[l])
        ws_ori[l] = scipy.stats.multivariate_normal.rvs(ws_ori[l], cov=.1)
        for i in range(2):
            thetas_ori[i, ls[i] == l] = sigmoid(np.dot(xs[i, ls[i] == l], ws_ori[l]))
            # xs[ls == l] += l
            ys[i, ls[i] == l] = scipy.stats.bernoulli.rvs(thetas_ori[i, ls[i] == l])

    return xs, ys, ls, gs, M_ori, ws_ori, thetas_ori

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_groups", "-l", type=int, default=4)
    parser.add_argument("--num_dims", "-d", type=int, default=5)
    parser.add_argument("--num_group_dims", "-k", type=int, default=3)
    parser.add_argument("--num_data", "-n", type=int, default=100)
    parser.add_argument("--num_iter", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=1.)
    parser.add_argument("--delta", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-dp", action='store_true', default=False)

    TRAIN = 0
    TEST = 1

    args = parser.parse_args()
    xs, ys, ls, gs, M_ori, ws_ori, thetas_ori = generate_data(
        L=args.num_groups, D=args.num_dims, K=args.num_group_dims, N=args.num_data
    )
    np.random.seed(args.seed)

    epsilon = None if args.no_dp else args.epsilon

    all_params, all_losses = infer(
        xs[TRAIN], ys[TRAIN], ls[TRAIN], gs, args.batch_size, args.num_iter, epsilon, args.delta, PRNGKey(args.seed)
    )
    # print(losses)
    # print(params)

    import sklearn.metrics as metrics

    def compute_roc(params):
        etas = gs @ params['M_loc'].T
        ps = jax.nn.sigmoid(np.einsum('nd,nd->n', xs[TEST], etas[ls[TEST]]))
        auc = metrics.roc_auc_score(ys[TEST], ps)
        return auc

    all_aucs = [compute_roc(params) for params in all_params]
    iter_axis = np.arange(len(all_aucs)) * 10

    # from sklearn.linear_model import LogisticRegression
    # baseline_model = LogisticRegression()
    # baseline_model.fit(xs[TRAIN], ys[TRAIN])
    # baseline_ps = baseline_model.predict_proba(xs[TEST])[:,-1]
    # baseline_auc = metrics.roc_auc_score(ys[TEST], baseline_ps)
    # print(f"Simple sklearn.LogReg AUC: {baseline_auc}")
    all_aucs = np.array([iter_axis, all_aucs]).T
    with open(f"results/logreg_{'nodp' if args.no_dp else f'eps_{args.epsilon}'}_n{args.num_data}_i{args.num_iter}_s{args.seed}.npz", 'wb') as f:
        np.savez(f, all_aucs)

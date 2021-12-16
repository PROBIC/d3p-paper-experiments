""" Alternative implementation for minibatch subsampling based on built-in jax routines
for comparison in Table 3. """


from d3p.util import is_int_scalar, is_array, example_count, sample_from_array
from numpyro.handlers import scale
import jax.numpy as jnp
import jax
import hashlib

from d3p.minibatch import q_to_batch_size


def subsample_batchify_data_builtin(dataset, batch_size=None, q=None, with_replacement=False):
    """Returns functions to fetch (randomized) batches of a given dataset by
    uniformly random subsampling using built-in jax routines instead of the Feistel shuffle.

    As `split_batchify_data`, takes the common epoch viewpoint to training,
    where an epoch is understood to be one pass over the data set. However,
    the data set is not shuffled and split to generate batches - instead
    every batch is drawn uniformly at random from the data set. An epoch thus
    merely refers to a number of batches that make up the same amount of data
    as the full data set.

    While each element of the data set in expectation occurs once per epoch,
    there are no guarantees to the exact number of appearances.

    The subsampling can be performed with or without replacement per batch.
    In the latter case (default), an element cannot occur more than once in a batch.

    The batches are guaranteed to always be of size batch_size. If the number of
    items in the data set is not evenly divisible by batch_size, the total number
    of elements contained in batches per epoch will be slightly less than the
    size of the data set.

    :param arrays: Tuple of arrays constituting the data set to be batchified.
        All arrays must have the same length on the first axis.
    :param batch_size: Size of the batches as absolute number. Mutually exclusive with q.
    :param q: Size of batches as ratio of the data set size. Mutually exlusive with batch_size.
    :return: tuple (init_fn: () -> (num_batches, batchifier_state), get_batch: (i, batchifier_state) -> batch)
        init_fn() returns the number of batches per epoch and an initialized state of the batchifier for the epoch
        get_batch() returns the next batch_size amount of items from the data set
    """
    if batch_size is None and q is None:
        raise ValueError("Either batch_size or batch ratio q must be given")
    if batch_size is not None and q is not None:
        raise ValueError("Only one of batch_size and batch ratio q must be given")
    if not dataset:
        raise ValueError("The data set must not be empty")

    num_records = example_count(dataset[0])
    for arr in dataset:
        if num_records != example_count(arr):
            raise ValueError("All arrays constituting the data set must have the same number of records")

    if batch_size is None:
        batch_size = q_to_batch_size(q, num_records)

    @jax.jit
    def init(rng_key):
        """ Initializes the batchifier for a new epoch.

        :param rng_key: The base PRNG key the batchifier will use for randomness.
        :return: tuple consisting of: number of batches in the epoch,
            initialized state of the batchifier for the epoch
        """
        return num_records // batch_size, rng_key

    @jax.jit
    def get_batch(i, batchifier_state):
        """ Fetches the next batch for the current epoch.

        :param i: The number of the batch in the epoch.
        :param batchifier_state: The initialized state returned by init.
        :return: the batch
        """
        rng_key = batchifier_state
        batch_rng_key = jax.random.fold_in(rng_key, i)
        ret_idx = jax.random.choice(batch_rng_key, jnp.arange(num_records), (batch_size,), replace=with_replacement)
        return tuple(jnp.take(a, ret_idx, axis=0) for a in dataset)

    return init, get_batch

if __name__ == '__main__':
    x = jnp.arange(100)
    y = -x
    x = jnp.expand_dims(x, -1) * jnp.ones(4)
    dataset = (x, y)
    init, get_batch = subsample_batchify_data_builtin(dataset, batch_size=60, with_replacement=True)

    rng_key = jax.random.PRNGKey(0)
    _, state = init(rng_key)
    batch_x, batch_y = get_batch(1, state)
    print(batch_x)
    print(jnp.sort(batch_y))
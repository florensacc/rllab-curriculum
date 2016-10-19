import tempfile

from sandbox.rocky.s3.resource_manager import resource_manager
import numpy as np
import multiprocessing


def calc_n_trajs(len_pairs, buckets):
    bins = np.zeros((len_pairs.shape[0],))
    for bucket in buckets:
        bins += np.all(np.less_equal(len_pairs, bucket), axis=1)
    bins -= 1
    n_trajs = np.zeros((len(buckets),))
    for i in range(len(buckets)):
        n_trajs[i] = np.sum(bins == i)
    return n_trajs


def eval_buckets(len_pairs, buckets):
    # return the variance of the number of trajs among all buckets
    return np.var(calc_n_trajs(len_pairs, buckets))


def compute_one_sol(args):
    len_pairs, n_buckets, seed = args
    np.random.seed(seed)
    max_len = np.max(len_pairs)
    bins = np.linspace(0, max_len, n_buckets + 1, dtype=np.int)[1:]
    buckets = np.asarray(list(zip(bins, bins)))

    for idx in range(100):
        # print("here", idx)
        candidates = [(eval_buckets(len_pairs, buckets), buckets)]

        all_buckets_eps = buckets[np.newaxis, :, :] + \
                         np.random.uniform(low=-5, high=5, size=(100,) + buckets.shape)
        all_buckets_eps = np.cast['int'](np.clip(all_buckets_eps, 1, max_len))

        satisfied = np.all(all_buckets_eps[:, 1:] >= all_buckets_eps[:, :-1], axis=(1, 2))

        all_buckets_eps = all_buckets_eps[satisfied]

        all_buckets_eps[:, -1] = max_len

        for buckets_eps in all_buckets_eps:
            candidates.append((eval_buckets(len_pairs, buckets_eps), buckets_eps))
        candidates = sorted(candidates, key=lambda x: x[0])
        sol, buckets = candidates[0]
    # print(sol, flush=True)
    return sol, buckets


def compute_optimal_bucket(len_pairs, n_buckets):
    with multiprocessing.Pool() as pool:
        results = pool.map(compute_one_sol, [(len_pairs, n_buckets, idx) for idx in range(100)])
    best_sol, best_bucket = sorted(results, key=lambda x: x[0])[0]
    return best_bucket


class BucketDataset(object):
    def __init__(self, data_dict, demo_cache_key, n_buckets=5):
        demo_paths = data_dict["demo_paths"]
        analogy_paths = data_dict["analogy_paths"]

        demo_lens = [len(p["rewards"]) for p in demo_paths]
        analogy_lens = [len(p["rewards"]) for p in analogy_paths]

        bucket_cache_key = "bucket-allocs-v1/" + demo_cache_key + "-" + "buckets" + str(n_buckets)

        def mkbucket():
            len_pairs = np.asarray(list(zip(demo_lens, analogy_lens)))
            best_bucket = compute_optimal_bucket(len_pairs, n_buckets)
            f = tempfile.NamedTemporaryFile()
            np.savez_compressed(f, bucket=best_bucket)
            resource_manager.register_file(bucket_cache_key, f.name)
            f.close()

        bucket_file = resource_manager.get_file(bucket_cache_key, mkbucket)

        print(bucket_file)

        bucket = np.load(bucket_file)["bucket"]

        # paths should be grouped by bucket, and each time we sample a minibatch from a single bucket
        import ipdb; ipdb.set_trace()


        pass


if __name__ == "__main__":
    resource_name = "simple-seq-2-100-full-state-v1-seq3-pts4"

    file_name = resource_manager.get_file(resource_name)
    data_dict = dict(np.load(file_name))
    dataset = BucketDataset(data_dict, demo_cache_key=resource_name)
    import ipdb;

    ipdb.set_trace()
    pass

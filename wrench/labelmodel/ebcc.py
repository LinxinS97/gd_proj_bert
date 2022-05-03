from typing import Optional, Any, Union
import numpy as np
import scipy.sparse as ssp
from scipy.special import digamma, gammaln
from scipy.stats import entropy, dirichlet

from wrench.basemodel import BaseLabelModel
from wrench.dataset import BaseDataset


def ebcc_vb(tuples,
            num_groups=10,  # M
            a_pi=0.1,
            alpha=1,  # alpha_k, it can be 1 or \sum_i gamma_ik
            a_v=4,  # beta_kk
            b_v=1,  # beta_kk', k neq k'

            seed=1234,
            max_iter=500,
            empirical_prior=False):

    num_items, num_workers, num_classes = tuples.max(axis=0) + 1
    num_labels = tuples.shape[0]

    y_is_one_lij = []
    y_is_one_lji = []
    for k in range(num_classes):
        selected = (tuples[:, 2] == k)
        coo_ij = ssp.coo_matrix((np.ones(selected.sum()), tuples[selected, :2].T), shape=(num_items, num_workers),
                                dtype=np.bool)
        y_is_one_lij.append(coo_ij.tocsr())
        y_is_one_lji.append(coo_ij.T.tocsr())

    beta_kl = np.eye(num_classes) * (a_v - b_v) + b_v

    # initialize z_ik, zg_ikm
    z_ik = np.zeros((num_items, num_classes))
    for l in range(num_classes):
        z_ik[:, [l]] += y_is_one_lij[l].sum(axis=-1)
    z_ik /= z_ik.sum(axis=-1, keepdims=True)
    z_ik[np.isnan(z_ik)] = 0

    if empirical_prior:
        alpha = z_ik.sum(axis=0)

    np.random.seed(seed)

    # zg_ikm = rho_ikm (paper ver)
    # ============================== Inference =====================================
    zg_ikm = np.random.dirichlet(np.ones(num_groups), z_ik.shape) * z_ik[:, :, None]
    for it in range(max_iter):
        # update rules 4
        eta_km = a_pi / num_groups + zg_ikm.sum(axis=0)

        # update rules 3
        nu_k = alpha + z_ik.sum(axis=0)

        # update rules 5
        mu_jkml = np.zeros((num_workers, num_classes, num_groups, num_classes)) + beta_kl[None, :, None, :]
        for l in range(num_classes):
            for k in range(num_classes):
                mu_jkml[:, k, :, l] += y_is_one_lji[l].dot(zg_ikm[:, k, :])

        # expectations
        Eq_log_pi_km = digamma(eta_km) - digamma(eta_km.sum(axis=-1, keepdims=True))
        Eq_log_tau_k = digamma(nu_k) - digamma(nu_k.sum())
        Eq_log_v_jkml = digamma(mu_jkml) - digamma(mu_jkml.sum(axis=-1, keepdims=True))

        zg_ikm[:] = Eq_log_pi_km[None, :, :] + Eq_log_tau_k[None, :, None]
        for l in range(num_classes):
            for k in range(num_classes):
                zg_ikm[:, k, :] += y_is_one_lij[l].dot(Eq_log_v_jkml[:, k, :, l])

        # update rules 1
        zg_ikm = np.exp(zg_ikm)
        zg_ikm /= zg_ikm.reshape(num_items, -1).sum(axis=-1)[:, None, None]

        last_z_ik = z_ik

        # update rules 2
        z_ik = zg_ikm.sum(axis=-1)

        if np.allclose(last_z_ik, z_ik, atol=1e-3):
            break

    ELBO = ((eta_km - 1) * Eq_log_pi_km).sum() + ((nu_k - 1) * Eq_log_tau_k).sum() + (
                (mu_jkml - 1) * Eq_log_v_jkml).sum()
    ELBO += dirichlet.entropy(nu_k)
    for k in range(num_classes):
        ELBO += dirichlet.entropy(eta_km[k])
    ELBO += (gammaln(mu_jkml) - (mu_jkml - 1) * digamma(mu_jkml)).sum()
    alpha0_jkm = mu_jkml.sum(axis=-1)
    ELBO += ((alpha0_jkm - num_classes) * digamma(alpha0_jkm) - gammaln(alpha0_jkm)).sum()
    ELBO += entropy(zg_ikm.reshape(num_items, -1).T).sum()
    return z_ik, ELBO

def create_tuples(dataset: Union[BaseDataset, np.ndarray]):
    ids = np.repeat(np.array(range(len(dataset))), len(dataset.weak_labels[0]))
    workers = np.repeat(
        np.array([i for i in range(len(dataset.weak_labels[0]))]), len(dataset.weak_labels)
    ).reshape(len(dataset.weak_labels[0]), -1).T.reshape(-1)
    classes = np.array(dataset.weak_labels).reshape(-1)

    tuples = np.vstack((ids, workers, classes))
    tuples = tuples[:, tuples[2, :] != -1]

    return tuples.T



class EBCC(BaseLabelModel):
    def __init__(self,
                 num_groups: Optional[int]=10,
                 a_pi: Optional[float]=0.1,
                 alpha: Optional[float]=1,
                 a_v: Optional[float]=4,
                 b_v: Optional[float] = 1,
                 inference_iter: Optional[int] = 1000,
                 empirical_prior=False,
                 **kwargs: Any):
        super().__init__()
        self.hyperparas = {
            'num_groups': num_groups,
            'a_pi': a_pi,
            'alpha': alpha,
            'a_v': a_v,
            'b_v': b_v,
            'empirical_prior': empirical_prior,
        }
        self.inference_iter = inference_iter
        self.seed = None

    def fit(self,
            dataset_train: Union[BaseDataset, np.ndarray],
            dataset_valid: Optional[Union[BaseDataset, np.ndarray]] = None,
            y_valid: Optional[np.ndarray] = None,
            n_class: Optional[int] = None,
            verbose: Optional[bool] = False,
            *args: Any,
            **kwargs: Any):
        train_tuples = create_tuples(dataset_train)
        max_elbo = float('-inf')
        self.seed = None
        for _ in range(self.inference_iter):
            seed = np.random.randint(1e8)
            prediction, elbo = ebcc_vb(train_tuples, seed=seed, **self.hyperparas)
            if elbo > max_elbo:
                self.seed = seed

    def predict_proba(self,
                      dataset: Union[BaseDataset, np.ndarray],
                      **kwargs: Any) -> np.ndarray:
        tuples = create_tuples(dataset)

        prediction, _ = ebcc_vb(tuples, seed=self.seed, **self.hyperparas)
        return prediction

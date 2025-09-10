import random
import numpy as np

from .z_min_var import z_min_var
from .mig import mig
from .sap import sap
from .modularity import modularity
from .dci import dci
from .irs import irs
from .min_suf import estimate_min_suf
from .accuracies import calc_accs


def calc_dis_metrics(results, n_samples):
    idxs = random.choices(range(results['y'].shape[0]), k=n_samples)
    y_cat = np.zeros((n_samples, results['y'].shape[1]))
    for i in range(results['y'].shape[1]):
        classes = list(set(results['y'][:,i].numpy()))
        y_cat[:,i] = np.array([classes.index(i) for i in results['y'][idxs,i][:,None].numpy()])
    z = results['z'][idxs].numpy()
    min_val, suf_val = estimate_min_suf(y_cat, z)
    factor_val = z_min_var(y_cat, z)
    mig_val = mig(y_cat, z)
    sap_val = sap(y_cat, z)
    modularity_val = modularity(y_cat, z)
    dci_val = dci(results['y'][idxs], z)
    irs_val = irs(results['y'][idxs], z)
    return {
        "factor_vae":       factor_val,
        "mig":              mig_val,
        "sap":              sap_val,
        "modularity":       modularity_val,
        "disentanglement":  dci_val[0],
        "completeness":     dci_val[1],
        "irs":              irs_val,
        "minimality":       min_val,
        "sufficiency":      suf_val
    }


def get_all_metrics(results, n_samples=2000):
    return {
        **calc_dis_metrics(results, n_samples), 
        **calc_accs(results, n_samples),
        **calc_accs(results, 100),
        **calc_accs(results, 1000),
        **calc_accs(results, 10000),
    }
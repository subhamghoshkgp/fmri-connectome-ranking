"""Descriptive evaluation: CV folds/repeats are not independent subjects."""
import itertools
import numpy as np
from sklearn.metrics import (roc_auc_score, accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)


def prediction_metrics(y, score, predicted):
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return dict(auc=float(roc_auc_score(y, score)),
                accuracy=float(accuracy_score(y, predicted)),
                balanced_accuracy=float(balanced_accuracy_score(y, predicted)),
                sensitivity=float(tp/(tp+fn)), specificity=float(tn/(tn+fp)),
                f1=float(f1_score(y, predicted, zero_division=0)))


def stability(supports, p):
    rows = []
    for i,j in itertools.combinations(range(len(supports)), 2):
        a,b = set(supports[i]), set(supports[j])
        k,l,r = len(a),len(b),len(a & b)
        # Empty/empty is undefined, never interpreted as perfect discovery.
        jac = r/len(a|b) if a|b else None
        kun = (r-k*k/p)/(k-k*k/p) if k==l and 0<k<p else None
        # Explicit descriptive extension for unequal sizes, not standard Kuncheva.
        den = min(k,l)-k*l/p
        adjusted = (r-k*l/p)/den if den>0 else None
        rows.append(dict(i=i,j=j,size_i=k,size_j=l,intersection=r,
                         jaccard=jac,kuncheva_equal_size=kun,
                         adjusted_overlap_min=adjusted))
    result = {}
    for key in ['jaccard','kuncheva_equal_size','adjusted_overlap_min']:
        values = [r[key] for r in rows if r[key] is not None]
        result[key] = float(np.mean(values)) if values else None
        result[key+'_valid_pairs'] = len(values)
    Z = np.zeros((len(supports),p))
    for i,s in enumerate(supports): Z[i,s] = 1
    q = Z.mean() if len(Z) else 0
    result['nogueira'] = (float(1-Z.var(axis=0,ddof=1).mean()/(q*(1-q)))
                          if len(Z)>1 and 0<q<1 else None)
    result['pairs'] = rows
    return result

"""Tests of the scikit-learn compatible estimators."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.estimator_checks import parametrize_with_checks

from svm_smo import SmoLSSVR, SmoSVC, SmoSVR

# `gamma="scale"` depends on the data variance, which changes when samples are repeated
_WEIGHT_EQUIVALENCE = {
    "check_sample_weight_equivalence_on_dense_data": "gamma='scale' is data dependent"
}


@parametrize_with_checks(
    [SmoSVR(), SmoLSSVR(), SmoSVC()],
    expected_failed_checks=lambda est: _WEIGHT_EQUIVALENCE,
)
def test_sklearn_compatible(estimator, check):
    check(estimator)


def test_svc_string_labels_and_grid_search():
    x, y = make_classification(200, 5, random_state=0)
    labels = np.array(["no", "yes"])[y]
    model = make_pipeline(StandardScaler(), SmoSVC())
    search = GridSearchCV(model, {"smosvc__lam": [1.0, 0.1, 0.01]}, cv=3).fit(x, labels)
    assert set(search.predict(x)) <= {"no", "yes"}
    assert search.best_score_ > 0.8


def test_svc_matches_libsvm_and_balanced_weights():
    x, y = make_classification(150, 4, weights=[0.8], random_state=1)
    ours = SmoSVC(lam=0.5, gamma=0.3, tol=1e-8).fit(x, y)
    assert ours.dual_coef_.shape == (1, ours.n_support_)
    ref = SVC(C=2.0, gamma=0.3, tol=1e-8).fit(x, y)
    np.testing.assert_allclose(ours.decision_function(x), ref.decision_function(x), atol=1e-5)
    assert ours.n_support_ == ref.n_support_.sum()
    bal = SmoSVC(lam=0.5, gamma=0.3, class_weight="balanced").fit(x, y)
    ref = SVC(C=2.0, gamma=0.3, class_weight="balanced").fit(x, y)
    assert np.mean(bal.predict(x) == ref.predict(x)) > 0.97


def test_multiclass_one_vs_rest():
    from sklearn.datasets import load_iris

    x, y = load_iris(return_X_y=True)
    est = SmoSVC(lam=0.01).fit(x, y)
    assert est.decision_function(x).shape == (150, 3)
    assert est.score(x, y) > 0.95


def test_pickle_roundtrip():
    import pickle

    x, y = make_classification(60, 5, random_state=3)
    est = SmoSVC().fit(x, y)
    again = pickle.loads(pickle.dumps(est))
    np.testing.assert_array_equal(est.decision_function(x), again.decision_function(x))


def test_lssvr_matches_kernel_ridge_without_bias():
    x, y = make_regression(100, 3, noise=0.1, random_state=0)
    y = y - y.mean()
    ours = SmoLSSVR(lam=0.1, gamma=0.2, tol=1e-10).fit(x, y)
    # the bias is small for centred data, so predictions are close to kernel ridge
    ref = KernelRidge(alpha=0.1, kernel="rbf", gamma=0.2).fit(x, y)
    assert np.corrcoef(ours.predict(x), ref.predict(x))[0, 1] > 0.99


def test_precomputed_estimator():
    x, y = make_classification(80, 5, random_state=2)
    from svm_smo import kernel_matrix

    k = kernel_matrix(x, x, gamma=0.5)
    a = SmoSVC(kernel="precomputed").fit(k, y)
    b = SmoSVC(gamma=0.5).fit(x, y)
    np.testing.assert_allclose(a.decision_function(k), b.decision_function(x), atol=1e-6)


def test_invalid_parameters():
    x, y = make_classification(30, 5, random_state=0)
    with pytest.raises(ValueError):
        SmoSVC(lam=0).fit(x, y)
    with pytest.raises(ValueError):
        SmoSVC(gamma=-1.0).fit(x, y)
    est = clone(SmoSVR(epsilon=0.3))
    assert est.get_params()["epsilon"] == 0.3

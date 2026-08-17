"""
Load *exported* target models (output/model/) for the Shokri MIA, instead of
re-training them.

Every dataset directory follows the same layout::

    <DATASET>/<FamilyDir>/output/model/std_<fam>_model.<ext>
    <DATASET>/<FamilyDir>/output/model/dp_<fam>_model_eps_<eps>.<ext>

so a single mapping (``FAMILIES``) drives loading for LUNG_CANCER and every
other dataset in the study.

Three things make a saved target usable by ``shokri_mia``:

1. **Unpickling.** The DP-SVM targets were pickled from the shared
   ``common_svm`` module at the repo root, so that module must be importable
   (this file puts the repo root on ``sys.path``). The DNN objects were pickled
   from notebook ``__main__`` namespaces, so ``DNNClassifier`` /
   ``DPDNNClassifier`` are redefined below and registered into ``__main__`` so
   ``torch.load`` can resolve them.
2. **A ``predict_proba`` head.** The attack only reads posterior vectors. The
   DP-SVM exposes ``decision_function`` only and the DNNs return logits, so
   thin wrappers convert those to class posteriors (softmax).
3. **A matching feature space.** Each family is attacked in the same space it
   was trained in. The LR and SVM notebooks clip every row to unit L2 norm (so
   ``||x|| <= 1``); ``family_dataview`` reproduces that. The other families
   train directly on the MinMax-scaled features.

The module also exposes ``make_shadow`` factories that build *fresh* estimators
of the same kind as the target (shadow models are always trained from scratch,
which is inherent to the shadow-model attack).
"""

import os
import pickle
import sys

import numpy as np
from diffprivlib.models import GaussianNB as DPGaussianNB
from diffprivlib.models import LogisticRegression as DPLogReg
from diffprivlib.models import RandomForestClassifier as DPRandomForest
from scipy.special import expit, softmax
from sklearn.ensemble import RandomForestClassifier as SkRandomForest
from sklearn.linear_model import LogisticRegression as SkLogReg
from sklearn.naive_bayes import GaussianNB as SkGaussianNB
from sklearn.svm import SVC

# The DP-SVM implementation is shared with the training notebooks and lives at
# the repo root. Importing it here (rather than keeping a local copy) is what
# makes the shadow models identical to the targets they are calibrating.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from common_svm import DifferentiallyPrivateSVM, DPSVMOneVsRest  # noqa: E402

# --------------------------------------------------------------------------- #
# Family -> on-disk layout
# --------------------------------------------------------------------------- #
# (family dir under <DATASET>/, file-name stem, file extension)
FAMILIES = {
    "LR": ("LR", "lr", ".pkl"),
    "RF": ("RandomForest", "rf", ".pkl"),
    "GNB": ("GaussianNB", "gnb", ".pkl"),
    "SVM": ("SVM", "svm", ".pkl"),
    "DNN": ("DNN", "dnn", ".pt"),
}
MODEL_FAMILIES = list(FAMILIES)


def _eps_tag(eps):
    """Match the notebook export convention: 1.0 -> '1.0', 0.1 -> '0.1'."""
    return f"{float(eps):g}.0" if float(eps).is_integer() else f"{float(eps):g}"


def model_path(repo, dataset, family, variant, epsilon=None):
    fam_dir, stem, ext = FAMILIES[family]
    base = os.path.join(repo, dataset, fam_dir, "output", "model")
    if variant == "standard":
        primary = os.path.join(base, f"std_{stem}_model{ext}")
    else:
        primary = os.path.join(base, f"dp_{stem}_model_eps_{_eps_tag(epsilon)}{ext}")
    if os.path.exists(primary):
        return primary
    alt_ext = ".pkl" if ext == ".pt" else ".pt"
    alt = primary[: -len(ext)] + alt_ext
    if os.path.exists(alt):
        return alt
    return primary


def model_exists(repo, dataset, family, variant, epsilon=None):
    return os.path.exists(model_path(repo, dataset, family, variant, epsilon))


# --------------------------------------------------------------------------- #
# Custom classes so pickled targets resolve.  The DP-SVM classes come straight
# from ``common_svm`` (imported above) -- the same objects the notebooks trained
# and pickled -- so no local copy can drift out of sync with them.
# --------------------------------------------------------------------------- #
def _torch_dnn():
    import torch.nn as nn

    class DNNClassifier(nn.Module):
        def __init__(
            self, input_size, hidden_sizes=[128, 64], num_classes=3, dropout=0.3
        ):
            super().__init__()
            layers = []
            prev = input_size
            for h in hidden_sizes:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, num_classes))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    # DP variant is structurally identical; kept as a distinct name so the
    # opacus-wrapped pickle resolves its inner module class.
    class DPDNNClassifier(DNNClassifier):
        pass

    return DNNClassifier, DPDNNClassifier


# Register custom classes into __main__ so pickle/torch.load can resolve the
# qualified names they were saved under (notebook __main__).
def _register_for_unpickling():
    main = sys.modules.get("__main__")
    names = {
        "DifferentiallyPrivateSVM": DifferentiallyPrivateSVM,
        "DPSVMOneVsRest": DPSVMOneVsRest,
    }
    try:
        DNNClassifier, DPDNNClassifier = _torch_dnn()
        names["DNNClassifier"] = DNNClassifier
        names["DPDNNClassifier"] = DPDNNClassifier
    except Exception:
        pass  # torch missing -> DNN simply unavailable
    for name, obj in names.items():
        # expose on this module and on __main__
        setattr(sys.modules[__name__], name, obj)
        if main is not None and not hasattr(main, name):
            setattr(main, name, obj)


_register_for_unpickling()


# --------------------------------------------------------------------------- #
# predict_proba wrappers
# --------------------------------------------------------------------------- #
class _SVMProba:
    """Add a softmax posterior head to a (DP or linear) SVM target."""

    def __init__(self, model, n_classes):
        self._m = model
        self.n_classes = n_classes
        cls = getattr(model, "classes_", None)
        self.classes_ = np.asarray(cls if cls is not None else np.arange(n_classes))

    def _scores(self, X):
        # 2-D already for DPSVMOneVsRest / multiclass SVC (one column per
        # class); 1-D for a binary margin, which becomes [-s, s].
        s = self._m.decision_function(np.asarray(X, float))
        return s if s.ndim > 1 else np.column_stack([-s, s])

    def predict_proba(self, X):
        return softmax(self._scores(X), axis=1)

    def predict(self, X):
        return self.classes_[np.argmax(self._scores(X), axis=1)]


class _TorchProba:
    """Add a softmax posterior head to a torch DNN (raw or opacus-wrapped)."""

    def __init__(self, module, n_classes):
        import torch

        self._torch = torch
        module.eval()
        self._m = module
        self.n_classes = n_classes
        self.classes_ = np.arange(n_classes)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        with self._torch.no_grad():
            logits = self._m(self._torch.from_numpy(X))
            p = self._torch.softmax(logits, dim=1).cpu().numpy()
        return p

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


class _TorchShadowDNN:
    """A trainable, non-private DNN shadow matching the target architecture.

    What the shadow trains on depends on the caller: the Shokri attack
    (``shokri_mia``) fits shadows on a small synthetic pool, while LiRA
    (``lira.py``) fits them on real random halves of the dataset. Either way the
    shadow mirrors the target's architecture and optimisation -- hidden sizes,
    dropout, epoch count, and the target's minibatch rule -- so its IN/OUT
    confidence spread describes the target being calibrated. It stays
    non-private even against DP-DNN targets, whose flat posteriors are what the
    attack actually probes.
    """

    def __init__(
        self,
        n_features,
        n_classes,
        hidden=(64, 32),
        dropout=0.3,
        epochs=50,
        lr=1e-3,
        batch_size=None,
        seed=0,
    ):
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        # None -> use the DP notebooks' rule min(256, max(16, n // 8)); an int
        # pins the fixed batch size the STD notebook used for this dataset.
        self.batch_size = batch_size
        self.seed = seed
        self.classes_ = np.arange(n_classes)
        self._net = None

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.seed)
        DNNClassifier, _ = _torch_dnn()
        net = DNNClassifier(
            self.n_features, list(self.hidden), self.n_classes, self.dropout
        )
        Xt = torch.tensor(np.asarray(X), dtype=torch.float32)
        yt = torch.tensor(np.asarray(y), dtype=torch.long)
        n = Xt.shape[0]
        batch = self.batch_size if self.batch_size else min(256, max(16, n // 8))
        batch = max(1, min(batch, n))
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch, shuffle=True)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        net.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(net(xb), yb)
                loss.backward()
                opt.step()
        net.eval()
        self._net = net
        return self

    def predict_proba(self, X):
        import torch

        with torch.no_grad():
            logits = self._net(torch.tensor(np.asarray(X), dtype=torch.float32))
            return torch.softmax(logits, dim=1).numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


# --------------------------------------------------------------------------- #
# Feature-space view per family
# --------------------------------------------------------------------------- #
def _row_clip(X):
    """Scale down any row with ||x||_2 > 1; leave shorter rows untouched."""
    X = np.asarray(X, dtype=float)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1.0)


def family_dataview(family, data, low, high):
    """Return (data, low, high) in the space the family was trained in.

    LR and SVM clip every row to unit L2 norm (||x|| <= 1); other families use
    the MinMax-scaled features as-is. The clip is not a linear map, so the
    per-feature box is reported as the conservative [-1, 1] intersection --
    every coordinate of a unit-norm row lies inside it.
    """
    if family not in ("LR", "SVM"):
        return data, np.asarray(low, float), np.asarray(high, float)

    view = {
        "X_train": _row_clip(data["X_train"]),
        "X_test": _row_clip(data["X_test"]),
        "y_train": data["y_train"],
        "y_test": data["y_test"],
    }
    return (
        view,
        np.clip(np.asarray(low, float), -1.0, 1.0),
        np.clip(np.asarray(high, float), -1.0, 1.0),
    )


# --------------------------------------------------------------------------- #
# Target loading
# --------------------------------------------------------------------------- #
def load_target(repo, dataset, family, variant, epsilon, n_classes):
    """Load and wrap an exported target so it exposes predict_proba/predict."""
    path = model_path(repo, dataset, family, variant, epsilon)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if family == "DNN":
        import torch

        module = torch.load(path, map_location="cpu", weights_only=False)
        return _TorchProba(module, n_classes)

    with open(path, "rb") as f:
        model = pickle.load(f)

    if family == "SVM":
        return _SVMProba(model, n_classes)
    return model  # LR / RF / GNB already expose predict_proba


class _SignLabelDPSVM(DifferentiallyPrivateSVM):
    """``DifferentiallyPrivateSVM`` that accepts the dataset's own class labels.

    Chaudhuri's Algorithm 2 is stated for signed labels, so the binary notebooks
    map ``y`` to {-1, +1} (``y_train_svm = 2 * y_train - 1``) before calling
    ``fit``. Shadow models are handed the raw 0/1 labels, so the same conversion
    has to happen here -- otherwise the shadows optimise a different objective
    than the target and their IN/OUT confidence distributions are meaningless.
    """

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        # Highest label is the positive class, matching 2*y - 1 for y in {0, 1}.
        return super().fit(X, np.where(y == self.classes_[-1], 1, -1))


# --------------------------------------------------------------------------- #
# Per-dataset DNN target configs (read from each STD_DNN/DP_DNN notebook).
# A DNN shadow must mirror the target it calibrates, and the six targets differ
# in hidden sizes, epoch count, dropout, and batch rule -- so the shadow config
# is looked up per dataset and per variant. ``batch`` is a fixed int (the STD
# notebook's DataLoader batch) or "rule" (the DP notebooks' n-dependent rule,
# min(256, max(16, n_train // 8)), resolved at fit time).
# --------------------------------------------------------------------------- #
_DNN_CONFIGS = {
    "GALLSTONE": {
        "hidden": (64, 32),
        "lr": 1e-3,
        "standard": {"dropout": 0.3, "epochs": 50, "batch": 32},
        "dp": {"dropout": 0.3, "epochs": 50, "batch": "rule"},
    },
    "BCP": {
        "hidden": (64, 32),
        "lr": 1e-3,
        "standard": {"dropout": 0.2, "epochs": 50, "batch": 32},
        "dp": {"dropout": 0.3, "epochs": 50, "batch": "rule"},
    },
    "CANCER_RISK": {
        "hidden": (64, 32),
        "lr": 1e-3,
        "standard": {"dropout": 0.3, "epochs": 50, "batch": 32},
        "dp": {"dropout": 0.3, "epochs": 50, "batch": "rule"},
    },
    "KIDNEY_STONE": {
        "hidden": (64, 32),
        "lr": 1e-3,
        "standard": {"dropout": 0.3, "epochs": 50, "batch": 32},
        "dp": {"dropout": 0.3, "epochs": 50, "batch": "rule"},
    },
    "LUNG_CANCER": {
        "hidden": (128, 64),
        "lr": 1e-3,
        "standard": {"dropout": 0.3, "epochs": 50, "batch": 256},
        "dp": {"dropout": 0.3, "epochs": 30, "batch": "rule"},
    },
    "DIABETES": {
        "hidden": (64, 32),
        "lr": 1e-3,
        "standard": {"dropout": 0.3, "epochs": 50, "batch": 256},
        "dp": {"dropout": 0.3, "epochs": 50, "batch": "rule"},
    },
}
# Fallback for an unknown dataset name: a generic (64,32) net using the DP rule.
_DNN_CONFIG_DEFAULT = {
    "hidden": (64, 32),
    "lr": 1e-3,
    "standard": {"dropout": 0.3, "epochs": 50, "batch": "rule"},
    "dp": {"dropout": 0.3, "epochs": 50, "batch": "rule"},
}


def _dnn_shadow_kwargs(dataset, variant):
    """Resolve the ``_TorchShadowDNN`` kwargs for a (dataset, variant) target."""
    cfg = _DNN_CONFIGS.get(dataset, _DNN_CONFIG_DEFAULT)
    v = cfg["standard"] if variant == "standard" else cfg["dp"]
    batch = v["batch"]
    return dict(
        hidden=cfg["hidden"],
        lr=cfg["lr"],
        dropout=v["dropout"],
        epochs=v["epochs"],
        batch_size=(None if batch == "rule" else batch),
    )


# --------------------------------------------------------------------------- #
# Shadow factories (fresh estimators of the same kind as the target)
# --------------------------------------------------------------------------- #
def make_shadow_factory(
    family, variant, *, epsilon, bounds, n_features, n_classes, base_seed, dataset=None
):
    """Return a zero-arg callable producing a fresh shadow estimator."""
    counter = {"n": 0}

    def factory():
        counter["n"] += 1
        seed = base_seed * 100 + counter["n"]

        if family == "LR":
            if variant == "standard":
                return SkLogReg(max_iter=1000, random_state=seed)
            # Shadows see the row-clipped view (family_dataview), so ||x|| <= 1.
            return DPLogReg(
                epsilon=epsilon, max_iter=1000, random_state=seed, data_norm=1.0
            )

        if family == "RF":
            if variant == "standard":
                # Match the capacity-matched STD_RF target (20 trees, depth 4).
                return SkRandomForest(
                    n_estimators=20, max_depth=4, random_state=seed, n_jobs=-1
                )
            return DPRandomForest(
                n_estimators=20,
                epsilon=epsilon,
                random_state=seed,
                n_jobs=-1,
                max_depth=4,
                bounds=bounds,
            )

        if family == "GNB":
            if variant == "standard":
                return SkGaussianNB()
            # random_state is passed so DP-GNB draws reproducibly per seed; without
            # it diffprivlib falls back to the global RNG and seed replication in
            # the Yeom/LiRA drivers would not be reproducible.
            return DPGaussianNB(epsilon=epsilon, bounds=bounds, random_state=seed)

        if family == "SVM":
            if variant == "standard":
                # RBF, matching the STD_SVM notebooks. A linear shadow against
                # an RBF target would have a different hypothesis space, so its
                # IN/OUT confidence spread would not describe the target's.
                # ``probability`` stays off: the Platt head it fits is unused
                # (_SVMProba scores from decision_function, which probability=True
                # does not change) and it costs an internal 5-fold CV per shadow.
                svc = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed)
                return _FittableSVMProba(svc, n_classes)
            # Mirror how the notebooks trained the target: a binary target is a
            # single DifferentiallyPrivateSVM spending the whole budget, while a
            # multi-class target is One-vs-Rest, which splits epsilon across the
            # classes. Using OvR for both would train binary shadows at eps/2 and
            # leave them systematically noisier than the target they calibrate.
            if n_classes <= 2:
                dp = _SignLabelDPSVM(
                    epsilon=epsilon,
                    Lambda=0.01,
                    h=0.5,
                    normalize=True,
                    random_state=seed,
                )
            else:
                dp = DPSVMOneVsRest(
                    epsilon=epsilon,
                    Lambda=0.01,
                    h=0.5,
                    normalize=True,
                    random_state=seed,
                )
            return _FittableSVMProba(dp, n_classes)

        if family == "DNN":
            # Architecture is per-dataset and per-variant (see _DNN_CONFIGS);
            # the shadow stays non-private regardless of variant.
            kw = _dnn_shadow_kwargs(dataset, variant)
            return _TorchShadowDNN(n_features, n_classes, seed=seed, **kw)

        raise ValueError(family)

    return factory


class _FittableSVMProba(_SVMProba):
    """``_SVMProba`` that also trains its wrapped estimator (for shadows)."""

    def __init__(self, model, n_classes):
        self._raw = model
        self.n_classes = n_classes
        self.classes_ = np.arange(n_classes)

    def fit(self, X, y):
        self._raw.fit(np.asarray(X, float), np.asarray(y))
        self._m = self._raw
        cls = getattr(self._raw, "classes_", None)
        self.classes_ = np.asarray(
            cls if cls is not None else np.arange(self.n_classes)
        )
        return self

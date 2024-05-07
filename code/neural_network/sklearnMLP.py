import numpy as np

_STOCHASTIC_SOLVERS = ["sgd", "adam"]

def _fit(self, X, y, incremental=False):
    # Make sure self.hidden_layer_sizes is a list
    hidden_layer_sizes = self.hidden_layer_sizes
    if not hasattr(hidden_layer_sizes, "__iter__"):
        hidden_layer_sizes = [hidden_layer_sizes]
    hidden_layer_sizes = list(hidden_layer_sizes)

    if np.any(np.array(hidden_layer_sizes) <= 0):
        raise ValueError(
            "hidden_layer_sizes must be > 0, got %s." % hidden_layer_sizes
        )
    first_pass = not hasattr(self, "coefs_") or (
            not self.warm_start and not incremental
    )

    X, y = self._validate_input(X, y, incremental, reset=first_pass)

    n_samples, n_features = X.shape

    # Ensure y is 2D
    if y.ndim == 1:
        y = y.reshape((-1, 1))

    self.n_outputs_ = y.shape[1]

    layer_units = [n_features] + hidden_layer_sizes + [self.n_outputs_]

    # check random state
    self._random_state = check_random_state(self.random_state)

    if first_pass:
        # First time training the model
        self._initialize(y, layer_units, X.dtype)

    # Initialize lists
    activations = [X] + [None] * (len(layer_units) - 1)
    deltas = [None] * (len(activations) - 1)

    coef_grads = [
        np.empty((n_fan_in_, n_fan_out_), dtype=X.dtype)
        for n_fan_in_, n_fan_out_ in zip(layer_units[:-1], layer_units[1:])
    ]

    intercept_grads = [
        np.empty(n_fan_out_, dtype=X.dtype) for n_fan_out_ in layer_units[1:]
    ]

    # Run the Stochastic optimization solver
    if self.solver in _STOCHASTIC_SOLVERS:
        self._fit_stochastic(
            X,
            y,
            activations,
            deltas,
            coef_grads,
            intercept_grads,
            layer_units,
            incremental,
        )

    # Run the LBFGS solver
    elif self.solver == "lbfgs":
        self._fit_lbfgs(
            X, y, activations, deltas, coef_grads, intercept_grads, layer_units
        )

    # validate parameter weights
    weights = chain(self.coefs_, self.intercepts_)
    if not all(np.isfinite(w).all() for w in weights):
        raise ValueError(
            "Solver produced non-finite parameter weights. The input data may"
            " contain large values and need to be preprocessed."
        )

    return self
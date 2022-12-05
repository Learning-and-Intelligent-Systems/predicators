"""This code was taken from scikit-learn GMM implementation, version 0.16.1, and adapted to the linear Gaussian case"""

"""
Gaussian Mixture Models.

This implementation corresponds to frequentist (non-Bayesian) formulation
of Gaussian Mixture Models.
"""

# Author: Ron Weiss <ronweiss@gmail.com>
#         Fabian Pedregosa <fabian.pedregosa@inria.fr>
#         Bertrand Thirion <bertrand.thirion@inria.fr>

import warnings
import numpy as np
from scipy import linalg

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
# from sklearn.utils.extmath import logsumexp
from scipy.special import logsumexp
from sklearn.utils.validation import check_is_fitted
import sklearn
import sklearn.linear_model
import sklearn.cluster
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# from sklearn.externals.six.moves import zip

EPS = np.finfo(float).eps


def log_multivariate_normal_density(X, means, covars, covariance_type='diag'):
    """Compute the log probability under a multivariate Gaussian distribution.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        List of n_features-dimensional data points.  Each row corresponds to a
        single data point.
    means : array_like, shape (n_components, n_features)
        List of n_features-dimensional mean vectors for n_components Gaussians.
        Each row corresponds to a single mean vector.
    covars : array_like
        List of n_components covariance parameters for each Gaussian. The shape
        depends on `covariance_type`:
            (n_components, n_features)      if 'spherical',
            (n_features, n_features)    if 'tied',
            (n_components, n_features)    if 'diag',
            (n_components, n_features, n_features) if 'full'
    covariance_type : string
        Type of the covariance parameters.  Must be one of
        'spherical', 'tied', 'diag', 'full'.  Defaults to 'diag'.

    Returns
    -------
    lpr : array_like, shape (n_samples, n_components)
        Array containing the log probabilities of each data point in
        X under each of the n_components multivariate Gaussian distributions.
    """
    log_multivariate_normal_density_dict = {
        'spherical': _log_multivariate_normal_density_spherical,
        'tied': _log_multivariate_normal_density_tied,
        'diag': _log_multivariate_normal_density_diag,
        'full': _log_multivariate_normal_density_full}
    return log_multivariate_normal_density_dict[covariance_type](
        X, means, covars)


def sample_gaussian(mean, covar, covariance_type='diag', n_samples=1,
                    rng=None):
    """Generate random samples from a Gaussian distribution.

    Parameters
    ----------
    mean : array_like, shape (n_features,)
        Mean of the distribution.

    covar : array_like, optional
        Covariance of the distribution. The shape depends on `covariance_type`:
            scalar if 'spherical',
            (n_features) if 'diag',
            (n_features, n_features)  if 'tied', or 'full'

    covariance_type : string, optional
        Type of the covariance parameters.  Must be one of
        'spherical', 'tied', 'diag', 'full'.  Defaults to 'diag'.

    n_samples : int, optional
        Number of samples to generate. Defaults to 1.

    Returns
    -------
    X : array, shape (n_features, n_samples)
        Randomly generated sample
    """
    # rng = check_random_state(random_state)
    n_dim = mean.shape[1]
    rand = rng.randn(n_dim, n_samples)
    if n_samples == 1:
        rand.shape = (n_dim,)

    if covariance_type == 'spherical':
        rand *= np.sqrt(covar)
    elif covariance_type == 'diag':
        rand = np.dot(np.diag(np.sqrt(covar)), rand)
    else:
        s, U = linalg.eigh(covar)
        s.clip(0, out=s)        # get rid of tiny negatives
        np.sqrt(s, out=s)
        U *= s
        rand = np.dot(U, rand)

    return (rand.T + mean).T


class GMM(BaseEstimator):
    """Gaussian Mixture Model

    Representation of a Gaussian mixture model probability distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a GMM distribution.

    Initializes parameters such that every mixture component has zero
    mean and identity covariance.


    Parameters
    ----------
    n_components : int, optional
        Number of mixture components. Defaults to 1.

    covariance_type : string, optional
        String describing the type of covariance parameters to
        use.  Must be one of 'spherical', 'tied', 'diag', 'full'.
        Defaults to 'diag'.

    random_state: RandomState or an int seed (None by default)
        A random number generator instance

    min_covar : float, optional
        Floor on the diagonal of the covariance matrix to prevent
        overfitting.  Defaults to 1e-3.

    tol : float, optional
        Convergence threshold. EM iterations will stop when average
        gain in log-likelihood is below this threshold.  Defaults to 1e-3.

    n_iter : int, optional
        Number of EM iterations to perform.

    n_init : int, optional
        Number of initializations to perform. the best results is kept

    Attributes
    ----------
    weights_ : array, shape (`n_components`,)
        This attribute stores the mixing weights for each mixture component.

    means_ : array, shape (`n_components`, `n_features`)
        Mean parameters for each mixture component.

    covars_ : array
        Covariance parameters for each mixture component.  The shape
        depends on `covariance_type`::

            (n_components, n_features)             if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.



    See Also
    --------

    DPGMM : Infinite gaussian mixture model, using the dirichlet
        process, fit with a variational algorithm


    VBGMM : Finite gaussian mixture model fit with a variational
        algorithm, better for situations where there might be too little
        data to get a good estimate of the covariance matrix.

    Examples
    --------

    >>> import numpy as np
    >>> from sklearn import mixture
    >>> np.random.seed(1)
    >>> g = mixture.GMM(n_components=2)
    >>> # Generate random observations with two modes centered on 0
    >>> # and 10 to use for training.
    >>> obs = np.concatenate((np.random.randn(100, 1),
    ...                       10 + np.random.randn(300, 1)))
    >>> g.fit(obs) # doctest: +NORMALIZE_WHITESPACE
    GMM(covariance_type='diag', init_params='wmc', min_covar=0.001,
            n_components=2, n_init=1, n_iter=100, params='wmc',
            random_state=None, thresh=None, tol=0.001)
    >>> np.round(g.weights_, 2)
    array([ 0.75,  0.25])
    >>> np.round(g.means_, 2)
    array([[ 10.05],
           [  0.06]])
    >>> np.round(g.covars_, 2) #doctest: +SKIP
    array([[[ 1.02]],
           [[ 0.96]]])
    >>> g.predict([[0], [2], [9], [10]]) #doctest: +ELLIPSIS
    array([1, 1, 0, 0]...)
    >>> np.round(g.score([[0], [2], [9], [10]]), 2)
    array([-2.19, -4.58, -1.75, -1.21])
    >>> # Refit the model on new data (initial parameters remain the
    >>> # same), this time with an even split between the two modes.
    >>> g.fit(20 * [[0]] +  20 * [[10]]) # doctest: +NORMALIZE_WHITESPACE
    GMM(covariance_type='diag', init_params='wmc', min_covar=0.001,
            n_components=2, n_init=1, n_iter=100, params='wmc',
            random_state=None, thresh=None, tol=0.001)
    >>> np.round(g.weights_, 2)
    array([ 0.5,  0.5])

    """

    def __init__(self, n_components=1, covariance_type='diag',
                 random_state=None, thresh=None, tol=1e-3, min_covar=1e-3,
                 n_iter=100, n_init=1):
        if thresh is not None:
            warnings.warn("'thresh' has been replaced by 'tol' in 0.16 "
                          " and will be removed in 0.18.",
                          DeprecationWarning)
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.thresh = thresh
        self.tol = tol
        self.min_covar = min_covar
        self.random_state = random_state
        self.n_iter = n_iter
        self.n_init = n_init
        if covariance_type not in ['spherical', 'tied', 'diag', 'full']:
            raise ValueError('Invalid value for covariance_type: %s' %
                             covariance_type)

        if n_init < 1:
            raise ValueError('GMM estimation requires at least one run')
        if n_init > 1:
            raise ValueError('Current implementation only works for a single init')

        self.weights_ = np.ones(self.n_components) / self.n_components

        # flag to indicate exit status of fit() method: converged (True) or
        # n_iter reached (False)
        self.converged_ = False

    def _get_covars(self):
        """Covariance parameters for each mixture component.
        The shape depends on `cvtype`::

            (`n_states`, 'n_features')                if 'spherical',
            (`n_features`, `n_features`)              if 'tied',
            (`n_states`, `n_features`)                if 'diag',
            (`n_states`, `n_features`, `n_features`)  if 'full'
            """
        if self.covariance_type == 'full':
            return self.covars_
        elif self.covariance_type == 'diag':
            return [np.diag(cov) for cov in self.covars_]
        elif self.covariance_type == 'tied':
            return [self.covars_] * self.n_components
        elif self.covariance_type == 'spherical':
            return [np.diag(cov) for cov in self.covars_]

    def _set_covars(self, covars):
        """Provide values for covariance"""
        covars = np.asarray(covars)
        _validate_covars(covars, self.covariance_type, self.n_components)
        self.covars_ = covars

    def score_samples(self, X, y):
        """Return the per-sample likelihood of the data under the model.

        Compute the log probability of y under the model and
        return the posterior distribution (responsibilities) of each
        mixture component for each element of y.

        Parameters
        ----------
        y: array_like, shape (n_samples, n_labels)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        logprob : array_like, shape (n_samples,)
            Log probabilities of each data point in y.

        responsibilities : array_like, shape (n_samples, n_components)
            Posterior probabilities of each mixture component for each
            observation
        """
        check_is_fitted(self, 'regressor_')

        if X.ndim == 1:
            X = X[:, np.newaxis]
        if X.size == 0:
            return np.array([]), np.empty((0, self.n_components))
        if X.shape[1] != self.regressor_[0].coef_.shape[1]:
            raise ValueError('The shape of X  is not compatible with self')
        if y.shape[1] != self.regressor_[0].coef_.shape[0]:
            raise ValueError('The shape of y  is not compatible with self')

        means = np.empty((X.shape[0], self.n_components, self.regressor_[0].coef_.shape[0]))
        for k in range(self.n_components):
            means[:, k] = self.regressor_[k].predict(X)

        lpr = (log_multivariate_normal_density(y, means, self.covars_,
                                               self.covariance_type)
               + np.log(self.weights_))
        logprob = logsumexp(lpr, axis=1)
        responsibilities = np.exp(lpr - logprob[:, np.newaxis])
        return logprob, responsibilities

    def score(self, X, y):
        """Compute the log probability under the model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        logprob : array_like, shape (n_samples,)
            Log probabilities of each data point in X
        """
        logprob, _ = self.score_samples(X, y)
        return logprob

    def predict(self, X):
        """Predict label for data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = (n_samples,)
        """
        raise NotImplementedError('I think predict makes no sense for this formulation of linear GMM')
        """
        Note: the predict function of the traditional GMM seeks to get for a given X, what the correct
        cluster is. In the case of this implementation of the GMM, we'd need both X and y to make a 
        prediction of the most likely cluster, since we need X to compute the per-point cluster centroids
        and we need y to compute the likelihood of the data point given those cluster centroids. 
        Note that given X, what we can certainly do is sample y's under the GMM model, which is what we
        care about in this implementation.
        """
        logprob, responsibilities = self.score_samples(X)
        return responsibilities.argmax(axis=1)

    def predict_proba(self, X):
        """Predict posterior probability of data under each Gaussian
        in the model.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        responsibilities : array-like, shape = (n_samples, n_components)
            Returns the probability of the sample for each Gaussian
            (state) in the model.
        """
        raise NotImplementedError
        # Note: this is just like the predict(X) function
        logprob, responsibilities = self.score_samples(X)
        return responsibilities

    def predict_sample(self, x, rng=None):
        return self.sample(x[np.newaxis, :])[0]

    def sample(self, X, n_samples=1, rng=None):
        """Generate random samples from the model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
        n_samples : int, optional
            Number of samples to generate. Defaults to 1.

        Returns
        -------
        y : array_like, shape (n_samples, n_labels)
            List of samples
        """
        assert n_samples == 1, "This implementation returns as many samples as there are elements of X"
        check_is_fitted(self, 'regressor_')

        if rng is None:
            random_state = self.random_state
            rng = check_random_state(random_state)
        weight_cdf = np.cumsum(self.weights_)

        y = np.empty((X.shape[0], self.regressor_[0].coef_.shape[0]))
        rand = rng.rand(X.shape[0])
        # decide which component to use for each sample
        comps = weight_cdf.searchsorted(rand)
        # for each component, generate all needed samples
        for comp in range(self.n_components):
            # occurrences of current component in X
            comp_in_y = (comp == comps)
            # number of those occurrences
            num_comp_in_y = comp_in_y.sum()
            if num_comp_in_y > 0:
                if self.covariance_type == 'tied':
                    cv = self.covars_
                elif self.covariance_type == 'spherical':
                    cv = self.covars_[comp][0]
                else:
                    cv = self.covars_[comp]
                means = self.regressor_[comp].predict(X[comp_in_y])
                y[comp_in_y] = sample_gaussian(
                    means, cv, self.covariance_type,
                    num_comp_in_y, rng=rng).T
        return y

    def fit(self, X, y):
        """Estimate model parameters with the expectation-maximization
        algorithm.

        A initialization step is performed before entering the em
        algorithm. If you want to avoid this step, set the keyword
        argument init_params to the empty string '' when creating the
        GMM object. Likewise, if you would like just to do an
        initialization, set n_iter=0.

        Parameters
        ----------
        X : array_like, shape (n, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        y: array_like, shape (n, n_labels)
        """
        # initialization step
        assert X.shape[0] == y.shape[0]
        if X.shape[0] < self.n_components:
            raise ValueError(
                'GMM estimation with %s components, but got only %s samples' %
                (self.n_components, X.shape[0]))

        max_log_prob = -np.infty
        self.regressor_ = [sklearn.linear_model.Ridge(alpha=1e-3) 
                            for _ in range(self.n_components)]
        for _ in range(self.n_init):
            # Initialize via k-means and regress over the clusters to find closest fit
            cluster_idx = sklearn.cluster.KMeans(
                n_clusters=self.n_components,
                random_state=self.random_state).fit_predict(y)
            k = 0
            while k < self.n_components:
                current_cluster_mask = cluster_idx == k
                if sum(current_cluster_mask) > 0:
                    self.regressor_[k].fit(X[current_cluster_mask], y[current_cluster_mask])
                    k += 1
                else:
                    self.regressor_.pop(k)
                    self.n_components -= 1
                    # self.regressor_[k].coef_ = np.zeros((y.shape[1], X.shape[1]))
                    # self.regressor_[k].intercept_ = np.zeros(y.shape[1])
                    # self.regressor_[k].n_features_in = X.shape[1]

            self.weights_ = np.tile(1.0 / self.n_components,
                                    self.n_components)
            if y.shape[0] > 1:
                cv = np.cov(y.T) + self.min_covar * np.eye(y.shape[1])
            else:
                cv = self.min_covar + np.eye(y.shape[1])
            if not cv.shape:
                cv.shape = (1, 1)
            self.covars_ = \
                distribute_covar_matrix_to_match_covariance_type(
                    cv, self.covariance_type, self.n_components)

            # EM algorithms
            current_log_likelihood = None
            # reset self.converged_ to False
            self.converged_ = False

            # this line should be removed when 'thresh' is removed in v0.18
            tol = (self.tol if self.thresh is None
                   else self.thresh / float(X.shape[0]))

            for i in range(self.n_iter):
                prev_log_likelihood = current_log_likelihood
                # Expectation step
                log_likelihoods, responsibilities = self.score_samples(X, y)
                current_log_likelihood = log_likelihoods.mean()

                # Check for convergence.
                # (should compare to self.tol when dreprecated 'thresh' is
                # removed in v0.18)
                if prev_log_likelihood is not None:
                    change = abs(current_log_likelihood - prev_log_likelihood)
                    if change < tol:
                        self.converged_ = True
                        break

                # Maximization step
                self._do_mstep(X, y, responsibilities, self.min_covar)

            # if the results are better, keep it
            if self.n_iter:
                if current_log_likelihood > max_log_prob:
                    max_log_prob = current_log_likelihood
            #         best_params = {'weights': self.weights_,
            #                        'means': self.means_,
            #                        'covars': self.covars_}
        # check the existence of an init param that was not subject to
        # likelihood computation issue.
        if np.isneginf(max_log_prob) and self.n_iter:
            raise RuntimeError(
                "EM algorithm was never able to compute a valid likelihood " +
                "given initial parameters. Try different init parameters " +
                "(or increasing n_init) or check for degenerate data.")
        # # self.n_iter == 0 occurs when using GMM within HMM
        # if self.n_iter:
        #     self.covars_ = best_params['covars']
        #     self.means_ = best_params['means']
        #     self.weights_ = best_params['weights']
        return self

    def _do_mstep(self, X, y, responsibilities, min_covar=0):
        """ Perform the Mstep of the EM algorithm and return the class weihgts.
        """
        weights = responsibilities.sum(axis=0)
        weighted_y_sum = np.dot(responsibilities.T, X)
        inverse_weights = 1.0 / (weights[:, np.newaxis] + 10 * EPS)
        self.weights_ = (weights / (weights.sum() + 10 * EPS) + EPS)

        means = np.empty((X.shape[0], self.n_components, y.shape[1]))
        for k in range(self.n_components):
            self.regressor_[k].fit(X, y, responsibilities[:, k])
            means[:, k] = self.regressor_[k].predict(X)

        covar_mstep_func = _covar_mstep_funcs[self.covariance_type]
        self.covars_ = covar_mstep_func(
            means, y, responsibilities, weighted_y_sum, inverse_weights,
            min_covar)
        return weights

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        ndim_y = self.regressor[0].coef_.shape[0]
        ndim_X = self.regressor[0].coef_.shape[1]
        if self.covariance_type == 'full':
            cov_params = self.n_components * ndim_y * (ndim_y + 1) / 2.
        elif self.covariance_type == 'diag':
            cov_params = self.n_components * ndim_y
        elif self.covariance_type == 'tied':
            cov_params = ndim_y * (ndim_y + 1) / 2.
        elif self.covariance_type == 'spherical':
            cov_params = self.n_components
        mean_params = ndim_X * ndim_y * self.n_components
        return int(cov_params + mean_params + self.n_components - 1)

    def bic(self, X):
        """Bayesian information criterion for the current model fit
        and the proposed data

        Parameters
        ----------
        X : array of shape(n_samples, n_dimensions)

        Returns
        -------
        bic: float (the lower the better)
        """
        return (-2 * self.score(X).sum() +
                self._n_parameters() * np.log(X.shape[0]))

    def aic(self, X):
        """Akaike information criterion for the current model fit
        and the proposed data

        Parameters
        ----------
        X : array of shape(n_samples, n_dimensions)

        Returns
        -------
        aic: float (the lower the better)
        """
        return - 2 * self.score(X).sum() + 2 * self._n_parameters()

class IncrementalGMM(GMM):
    def fit(self, X, y):
        """Estimate model parameters with the expectation-maximization
        algorithm.

        A initialization step is performed before entering the em
        algorithm. If you want to avoid this step, set the keyword
        argument init_params to the empty string '' when creating the
        GMM object. Likewise, if you would like just to do an
        initialization, set n_iter=0.

        Parameters
        ----------
        X : array_like, shape (n, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        y: array_like, shape (n, n_labels)
        """
        # initialization step
        assert X.shape[0] == y.shape[0]
        if X.shape[0] < self.n_components:
            raise ValueError(
                'GMM estimation with %s components, but got only %s samples' %
                (self.n_components, X.shape[0]))

        max_log_prob = -np.infty
        # for _ in range(self.n_init):
        # Initialize via k-means and regress over the clusters to find closest fit
        if not hasattr(self, 'regressor_'):
            self.regressor_ = [sklearn.linear_model.Ridge(alpha=1e-3) 
                                for _ in range(self.n_components)]
            cluster_idx = sklearn.cluster.KMeans(
                n_clusters=self.n_components,
                random_state=self.random_state).fit_predict(y)
            k = 0
            while k < self.n_components:
                current_cluster_mask = cluster_idx == k
                if sum(current_cluster_mask) > 0:
                    self.regressor_[k].fit(X[current_cluster_mask], y[current_cluster_mask])
                    k += 1
                else:
                    self.regressor_.pop(k)
                    self.n_components -= 1
                    # self.regressor_[k].coef_ = np.zeros((y.shape[1], X.shape[1]))
                    # self.regressor_[k].intercept_ = np.zeros(y.shape[1])
                    # self.regressor_[k].n_features_in = X.shape[1]

            self.weights_ = np.tile(1.0 / self.n_components,
                                    self.n_components)
            if y.shape[0] > 1:
                cv = np.cov(y.T) + self.min_covar * np.eye(y.shape[1])
            else:
                cv = self.min_covar + np.eye(y.shape[1])
            if not cv.shape:
                cv.shape = (1, 1)
            self.covars_ = \
                distribute_covar_matrix_to_match_covariance_type(
                    cv, self.covariance_type, self.n_components)

            self.A_ = [np.zeros((X.shape[1], X.shape[1])) for _ in range(self.n_components)]
            self.b_ = [np.zeros((X.shape[1], y.shape[1])) for _ in range(self.n_components)]

            self._A_tmp = [np.zeros((X.shape[1], X.shape[1])) for _ in range(self.n_components)]
            self._b_tmp = [np.zeros((X.shape[1], y.shape[1])) for _ in range(self.n_components)]

            self.raw_weights_ = np.zeros(self.n_components)
            self._raw_weights_tmp = np.zeros(self.n_components)

            self._prev_covars = np.zeros_like(self.covars_) + self.min_covar



            # ###################
            # # TODO: REMOVE THESE; JUST FOR DEBUGGING
            # import copy
            # self._X = np.empty((0, X.shape[1]))
            # self._y = np.empty((0, y.shape[1]))
            # self._responsibilities = np.empty((0, self.n_components))
            # self._means = np.empty((0, self.n_components, y.shape[1]))
            # self._regressor_debug = copy.deepcopy(self.regressor_)
            # ###################

        # EM algorithms
        current_log_likelihood = None
        # reset self.converged_ to False
        self.converged_ = False

        # this line should be removed when 'thresh' is removed in v0.18
        tol = (self.tol if self.thresh is None
               else self.thresh / float(X.shape[0]))

        for i in range(self.n_iter):
            prev_log_likelihood = current_log_likelihood
            # Expectation step
            log_likelihoods, responsibilities = self.score_samples(X, y)
            current_log_likelihood = log_likelihoods.mean()

            # Check for convergence.
            # (should compare to self.tol when dreprecated 'thresh' is
            # removed in v0.18)
            if prev_log_likelihood is not None:
                change = abs(current_log_likelihood - prev_log_likelihood)
                if change < tol:
                    self.converged_ = True
                    break

            # Maximization step
            self._do_mstep(X, y, responsibilities, self.min_covar)

        self.A_ = self._A_tmp[:]
        self.b_ = self._b_tmp[:]
        self.raw_weights_ = self._raw_weights_tmp
        self._prev_covars = self.covars_
        # if the results are better, keep it
        if self.n_iter:
            if current_log_likelihood > max_log_prob:
                max_log_prob = current_log_likelihood
        #         best_params = {'weights': self.weights_,
        #                        'means': self.means_,
        #                        'covars': self.covars_}


        # ###############
        # # TODO: REMOVE THIS; DEBUGGING ONLY
        # self._X = np.r_[self._X, X]
        # self._y = np.r_[self._y, y]
        # self._responsibilities = np.r_[self._responsibilities, responsibilities]
        # means_tmp = np.empty((X.shape[0], self.n_components, y.shape[1]))
        # for k in range(self.n_components):
        #     means_tmp[:, k] = self._regressor_debug[k].predict(X)
        # self._means = np.r_[self._means, means_tmp]
        # ###############

        ########## INIT FOR LOOP USED TO END HERE

        # check the existence of an init param that was not subject to
        # likelihood computation issue.
        if np.isneginf(max_log_prob) and self.n_iter:
            raise RuntimeError(
                "EM algorithm was never able to compute a valid likelihood " +
                "given initial parameters. Try different init parameters " +
                "(or increasing n_init) or check for degenerate data.")
        # # self.n_iter == 0 occurs when using GMM within HMM
        # if self.n_iter:
        #     self.covars_ = best_params['covars']
        #     self.means_ = best_params['means']
        #     self.weights_ = best_params['weights']
        return self

    def _do_mstep(self, X, y, responsibilities, min_covar=0):
        """ Perform the Mstep of the EM algorithm and return the class weihgts.
        """
        weights = responsibilities.sum(axis=0)
        weighted_y_sum = np.dot(responsibilities.T, X)
        inverse_weights = 1.0 / ((self.raw_weights_ + weights)[:, np.newaxis] + 10 * EPS)
        prev_weights = self.raw_weights_[:, np.newaxis]
        self.weights_ = ((weights + self.raw_weights_) / ((weights + self.raw_weights_).sum() + 10 * EPS) + EPS)

        means = np.empty((X.shape[0], self.n_components, y.shape[1]))

        # ################
        # # TODO: REMOVE THIS; DEBUGGING ONLY
        # X_tmp = np.r_[self._X, X]
        # y_tmp = np.r_[self._y, y]
        # responsibilities_tmp = np.r_[self._responsibilities, responsibilities]
        # weights_tmp = responsibilities_tmp.sum(axis=0)
        # weighted_y_sum_tmp = None
        # inverse_weights_tmp = 1.0 / (weights_tmp[:, np.newaxis] + 10 * EPS)
        # weights_tmp = (weights / (weights.sum() + 10 * EPS) + EPS)

        # means_tmp = np.empty((X.shape[0], self.n_components, y_tmp.shape[1]))

        # assert np.allclose(weights_tmp, self.weights_)
        # assert np.allclose(inverse_weights, inverse_weights_tmp)
        # ################
        for k in range(self.n_components):
            A = (X.T * responsibilities[:, k]).dot(X)
            b = (X.T * responsibilities[:, k]).dot(y)
            coef = np.linalg.inv(self.A_[k] + A + self.regressor_[k].alpha * np.eye(X.shape[1])).dot(self.b_[k] + b)
            self.regressor_[k].coef_ = coef.T
            self.regressor_[k].intercept_[:] = 0.

            self._A_tmp[k] = self.A_[k] + A
            self._b_tmp[k] = self.b_[k] + b

            means[:, k] = self.regressor_[k].predict(X)

            # ###################
            # # TODO: REMOVE THIS; DEBUGGING ONLY
            # A_tmp = (X_tmp.T * responsibilities_tmp[:, k]).dot(X_tmp) 
            # b_tmp = (X_tmp.T * responsibilities_tmp[:, k]).dot(y_tmp)
            # coef = np.linalg.inv(A_tmp + self._regressor_debug[k].alpha * np.eye(X_tmp.shape[1])).dot(b_tmp)
            # self._regressor_debug[k].coef_ = coef.T
            # self._regressor_debug[k].intercept_[:] = 0.
            # means_tmp[:, k] = self._regressor_debug[k].predict(X)
            # try:
            #     assert np.allclose(self.regressor_[k].coef_, self._regressor_debug[k].coef_)
            # except:
            #     print(self.regressor_[k].coef_)
            #     print(self._regressor_debug[k].coef_)
            #     raise
            # ###################

        covar_mstep_func = _incremental_covar_mstep_funcs[self.covariance_type]
        self.covars_ = covar_mstep_func(
            means, y, responsibilities, weighted_y_sum, inverse_weights,
            self._prev_covars, prev_weights,  min_covar)

        # ####################
        # # TODO: REMOVE THIS; DEBUG ONLY
        # covar_mstep_func_tmp = _covar_mstep_funcs[self.covariance_type]
        # covars = covar_mstep_func_tmp(
        #     np.r_[self._means, means_tmp], y_tmp, responsibilities_tmp, weighted_y_sum_tmp, inverse_weights_tmp,
        #     min_covar)
        # try:
        #     assert np.allclose(covars, self.covars_, rtol=1e-3, atol=1e-4)
        # except:
        #     print(covars)
        #     print(self.covars_)
        #     raise
        # ####################

        self._raw_weights_tmp = weights + self.raw_weights_
        return weights

#########################################################################
## some helper routines
#########################################################################


def _log_multivariate_normal_density_diag(X, means, covars):
    """Compute Gaussian log-density at X for a diagonal model"""
    n_samples, n_dim = X.shape

    lpr = -0.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1)
                  + np.sum((means ** 2) / covars[np.newaxis, :, :], 2)
                  - 2 * np.einsum('ij,ikj->ik', X, (means / covars[np.newaxis, :, :]))
                  + np.dot(X ** 2, (1.0 / covars).T))
    return lpr


def _log_multivariate_normal_density_spherical(X, means, covars):
    """Compute Gaussian log-density at X for a spherical model"""
    cv = covars.copy()
    if covars.ndim == 1:
        cv = cv[:, np.newaxis]
    if covars.shape[1] == 1:
        cv = np.tile(cv, (1, X.shape[-1]))
    return _log_multivariate_normal_density_diag(X, means, cv)


def _log_multivariate_normal_density_tied(X, means, covars):
    """Compute Gaussian log-density at X for a tied model"""
    cv = np.tile(covars, (means.shape[0], 1, 1))
    return _log_multivariate_normal_density_full(X, means, cv)


def _log_multivariate_normal_density_full(X, means, covars, min_covar=1.e-7):
    """Log probability for full covariance matrices.
    """
    n_samples, n_dim = X.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))
    for c, (mu, cv) in enumerate(zip(means, covars)):
        try:
            cv_chol = linalg.cholesky(cv, lower=True)
        except linalg.LinAlgError:
            # The model is most probably stuck in a component with too
            # few observations, we need to reinitialize this components
            cv_chol = linalg.cholesky(cv + min_covar * np.eye(n_dim),
                                      lower=True)
        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = linalg.solve_triangular(cv_chol, (X - mu).T, lower=True).T
        log_prob[:, c] = - .5 * (np.sum(cv_sol ** 2, axis=1) +
                                 n_dim * np.log(2 * np.pi) + cv_log_det)

    return log_prob


def _validate_covars(covars, covariance_type, n_components):
    """Do basic checks on matrix covariance sizes and values
    """
    from scipy import linalg
    if covariance_type == 'spherical':
        if len(covars) != n_components:
            raise ValueError("'spherical' covars have length n_components")
        elif np.any(covars <= 0):
            raise ValueError("'spherical' covars must be non-negative")
    elif covariance_type == 'tied':
        if covars.shape[0] != covars.shape[1]:
            raise ValueError("'tied' covars must have shape (n_dim, n_dim)")
        elif (not np.allclose(covars, covars.T)
              or np.any(linalg.eigvalsh(covars) <= 0)):
            raise ValueError("'tied' covars must be symmetric, "
                             "positive-definite")
    elif covariance_type == 'diag':
        if len(covars.shape) != 2:
            raise ValueError("'diag' covars must have shape "
                             "(n_components, n_dim)")
        elif np.any(covars <= 0):
            raise ValueError("'diag' covars must be non-negative")
    elif covariance_type == 'full':
        if len(covars.shape) != 3:
            raise ValueError("'full' covars must have shape "
                             "(n_components, n_dim, n_dim)")
        elif covars.shape[1] != covars.shape[2]:
            raise ValueError("'full' covars must have shape "
                             "(n_components, n_dim, n_dim)")
        for n, cv in enumerate(covars):
            if (not np.allclose(cv, cv.T)
                    or np.any(linalg.eigvalsh(cv) <= 0)):
                raise ValueError("component %d of 'full' covars must be "
                                 "symmetric, positive-definite" % n)
    else:
        raise ValueError("covariance_type must be one of " +
                         "'spherical', 'tied', 'diag', 'full'")


def distribute_covar_matrix_to_match_covariance_type(
        tied_cv, covariance_type, n_components):
    """Create all the covariance matrices from a given template
    """
    if covariance_type == 'spherical':
        cv = np.tile(tied_cv.mean() * np.ones(tied_cv.shape[1]),
                     (n_components, 1))
    elif covariance_type == 'tied':
        cv = tied_cv
    elif covariance_type == 'diag':
        cv = np.tile(np.diag(tied_cv), (n_components, 1))
    elif covariance_type == 'full':
        cv = np.tile(tied_cv, (n_components, 1, 1))
    else:
        raise ValueError("covariance_type must be one of " +
                         "'spherical', 'tied', 'diag', 'full'")
    return cv


def _covar_mstep_diag(means, X, responsibilities, weighted_X_sum, norm,
                      min_covar):
    """Performing the covariance M step for diagonal cases"""
    avg_X2 = np.dot(responsibilities.T, X * X) * norm
    avg_means2 = np.einsum('ij,ijk->jk', responsibilities, means * means) * norm
    avg_X_means = np.einsum('ij,ijk->jk', responsibilities, np.einsum('ij,ikj->ikj', X, means)) * norm
    return avg_X2 - 2 * avg_X_means + avg_means2 + min_covar


def _covar_mstep_spherical(*args):
    """Performing the covariance M step for spherical cases"""
    cv = _covar_mstep_diag(*args)
    return np.tile(cv.mean(axis=1)[:, np.newaxis], (1, cv.shape[1]))


def _covar_mstep_full(gmm, X, responsibilities, weighted_X_sum, norm,
                      min_covar):
    """Performing the covariance M step for full cases"""
    # Eq. 12 from K. Murphy, "Fitting a Conditional Linear Gaussian
    # Distribution"
    raise NotImplementedError("dont know how")
    n_features = X.shape[1]
    cv = np.empty((gmm.n_components, n_features, n_features))
    for c in range(gmm.n_components):
        post = responsibilities[:, c]
        mu = gmm.means_[c]
        diff = X - mu
        with np.errstate(under='ignore'):
            # Underflow Errors in doing post * X.T are  not important
            avg_cv = np.dot(post * diff.T, diff) / (post.sum() + 10 * EPS)
        cv[c] = avg_cv + min_covar * np.eye(n_features)
    return cv


def _covar_mstep_tied(gmm, X, responsibilities, weighted_X_sum, norm,
                      min_covar):
    # Eq. 15 from K. Murphy, "Fitting a Conditional Linear Gaussian
    # Distribution"
    raise NotImplementedError("dont know how")
    avg_X2 = np.dot(X.T, X)
    avg_means2 = np.dot(gmm.means_.T, weighted_X_sum)
    out = avg_X2 - avg_means2
    out *= 1. / X.shape[0]
    out.flat[::len(out) + 1] += min_covar
    return out

def _incremental_covar_mstep_diag(means, X, responsibilities, weighted_X_sum, norm,
                      prev_covar, prev_inverse_norm, min_covar):
    """Performing the covariance M step for diagonal cases"""
    avg_X2 = np.dot(responsibilities.T, X * X) * norm
    avg_means2 = np.einsum('ij,ijk->jk', responsibilities, means * means) * norm
    avg_X_means = np.einsum('ij,ijk->jk', responsibilities, np.einsum('ij,ikj->ikj', X, means)) * norm
    return (prev_covar - min_covar) * prev_inverse_norm * norm + avg_X2 - 2 * avg_X_means + avg_means2 + min_covar


def _incremental_covar_mstep_spherical(*args):
    """Performing the covariance M step for spherical cases"""
    cv = _covar_mstep_diag(*args)
    return np.tile(cv.mean(axis=1)[:, np.newaxis], (1, cv.shape[1]))


def _incremental_covar_mstep_full(gmm, X, responsibilities, weighted_X_sum, norm,
                      min_covar):
    """Performing the covariance M step for full cases"""
    # Eq. 12 from K. Murphy, "Fitting a Conditional Linear Gaussian
    # Distribution"
    raise NotImplementedError("dont know how")
    n_features = X.shape[1]
    cv = np.empty((gmm.n_components, n_features, n_features))
    for c in range(gmm.n_components):
        post = responsibilities[:, c]
        mu = gmm.means_[c]
        diff = X - mu
        with np.errstate(under='ignore'):
            # Underflow Errors in doing post * X.T are  not important
            avg_cv = np.dot(post * diff.T, diff) / (post.sum() + 10 * EPS)
        cv[c] = avg_cv + min_covar * np.eye(n_features)
    return cv


def _incremental_covar_mstep_tied(gmm, X, responsibilities, weighted_X_sum, norm,
                      min_covar):
    # Eq. 15 from K. Murphy, "Fitting a Conditional Linear Gaussian
    # Distribution"
    raise NotImplementedError("dont know how")
    avg_X2 = np.dot(X.T, X)
    avg_means2 = np.dot(gmm.means_.T, weighted_X_sum)
    out = avg_X2 - avg_means2
    out *= 1. / X.shape[0]
    out.flat[::len(out) + 1] += min_covar
    return out


_covar_mstep_funcs = {'spherical': _covar_mstep_spherical,
                      'diag': _covar_mstep_diag,
                      'tied': _covar_mstep_tied,
                      'full': _covar_mstep_full,
                      }

_incremental_covar_mstep_funcs = {'spherical': _incremental_covar_mstep_spherical,
                      'diag': _incremental_covar_mstep_diag,
                      'tied': _incremental_covar_mstep_tied,
                      'full': _incremental_covar_mstep_full,
                      }

import inspect
from functools import partial
import numpy as np
import time
import warnings

__all__ = ["run_nested_sampler", "unit_cube_identity", "lnlike_of_unit_cube"]


def unit_cube_identity(unit_coords):
    """Return the unit cube point unchanged.

    This is the prior given to nautilus when ``nautilus_transform_in_likelihood``
    is set, because the prior transform then runs inside the likelihood. It is
    defined at module level so that it can be pickled.

    Parameters
    ----------
    unit_coords : ndarray of shape ``(ndim,)``
        Point in the unit cube.

    Returns
    -------
    unit_coords : ndarray of shape ``(ndim,)``
        The same point.
    """
    return unit_coords


def lnlike_of_unit_cube(unit_coords, prior_transform=None, likelihood_function=None):
    """Return the likelihood of a unit cube point.

    The point is mapped to parameter space with ``prior_transform`` and the
    result is passed to ``likelihood_function``. This gives the same value as
    applying the two functions one after the other in separate processes. It is
    defined at module level so that a ``functools.partial`` of it can be pickled
    and sent to pool workers.

    Parameters
    ----------
    unit_coords : ndarray of shape ``(ndim,)``
        Point in the unit cube.
    prior_transform : callable
        Map from the unit cube to parameter space, usually ``model.prior_transform``.
    likelihood_function : callable
        Likelihood of a parameter vector.

    Returns
    -------
    lnlike : float
        The return value of ``likelihood_function(prior_transform(unit_coords))``.
    """
    return likelihood_function(prior_transform(unit_coords))


def pocomc_sampler():
    """Return pocoMC's Sampler with state files that leave out the likelihood.

    pocoMC saves whole state to dill. Prospect's likelihood holds the model,
    the observations and the SPS, so it is not saved, and loaded state keeps
    the live one.
    """
    import pocomc as pc

    class Sampler(pc.Sampler):
        def save_state(self, path):
            keep, self.log_likelihood = self.log_likelihood, None

            try:
                super().save_state(path)
            finally:
                self.log_likelihood = keep

        def load_state(self, path):
            keep = self.log_likelihood
            super().load_state(path)
            self.log_likelihood = keep

    return Sampler


def run_nested_sampler(model,
                       likelihood_function,
                       nested_sampler="dynesty",
                       nested_nlive=1000,
                       nested_neff=1000,
                       verbose=False,
                       nautilus_transform_in_likelihood=False,
                       **kwargs):
    """We give a model -- parameter discription and prior transform -- and a
    likelihood function. We get back samples, weights, and likelihood values.

    Parameters
    ----------
    model : instance of the :py:class:`prospect.models.SpecModel`
        The model parameterization and parameter state.
    likelihood_function : callable
        Likelihood function
    nested_sampler : str
        One of "dynesty", "nautilus", "ultranest", "nestle" or "pocomc".
        pocoMC is sequential Monte Carlo, not nested sampling. It samples the
        unit cube with ``model.prior_transform`` inside the likelihood,
        ``nested_neff`` sets its ``n_total``, and ``nested_nlive`` is not used
        (pass ``n_effective`` and ``n_active`` instead).
    nested_live : int
        Number of live points.
    nested_neff : float
        Minimum effective sample size.
    verbose : bool
        Whether to output sampler progress.
    nautilus_transform_in_likelihood : bool (optional, default: False)
        Only for nautilus. If ``True``, nautilus gets the identity as its prior
        and a likelihood that applies ``model.prior_transform`` itself. The
        transform then runs in the pool workers instead of serially in the main
        process. The log likelihood of each unit cube point is unchanged, and so
        is a nautilus checkpoint file, so a run can be resumed with either
        setting. The returned ``points`` are in parameter space either way. The
        returned sampler object has the identity as its prior, so its own
        ``posterior()`` method gives unit cube points.

    Returns
    -------
    samples : 3-tuple of ndarrays (loc, logwt, loglike)
        Loctions, log-weights, and log-likelihoods for the samples

    obj : Object
        The sampling results object. This will depend on the nested sampler being used.
    """
    if verbose:
        print(f"running {nested_sampler} for {nested_neff} effective samples")

    if nautilus_transform_in_likelihood:
        if nested_sampler != "nautilus":
            raise ValueError("nautilus_transform_in_likelihood is only "
                             f"available for nautilus, not '{nested_sampler}'.")
        if kwargs.get("vectorized", False):
            raise ValueError("nautilus_transform_in_likelihood does not "
                             "support vectorized likelihoods.")

    go = time.time()

    # Initialize the sampler.
    if nested_sampler == 'nautilus':
        from nautilus import Sampler
        sampler_init = Sampler
        if nautilus_transform_in_likelihood:
            unit_likelihood = partial(lnlike_of_unit_cube,
                                      prior_transform=model.prior_transform,
                                      likelihood_function=likelihood_function)
            init_args = (unit_cube_identity, unit_likelihood)
        else:
            init_args = (model.prior_transform, likelihood_function)
        init_kwargs = dict(pass_dict=False, n_live=nested_nlive,
                           n_dim=model.ndim)
    elif nested_sampler == 'ultranest':
        from ultranest import ReactiveNestedSampler
        sampler_init = ReactiveNestedSampler
        init_args = (model.theta_labels(), likelihood_function,
                     model.prior_transform)
        init_kwargs = dict()
    elif nested_sampler == 'dynesty':
        from dynesty import DynamicNestedSampler
        sampler_init = DynamicNestedSampler
        init_args = (likelihood_function, model.prior_transform, model.ndim)
        init_kwargs = dict(nlive=nested_nlive)
    elif nested_sampler == 'nestle':
        import nestle
        init_kwargs = dict()
    elif nested_sampler == 'pocomc':
        from pocomc import Prior
        from scipy.stats import uniform
        sampler_init = pocomc_sampler()
        unit_likelihood = partial(lnlike_of_unit_cube,
                                  prior_transform=model.prior_transform,
                                  likelihood_function=likelihood_function)
        init_args = (Prior(model.ndim * [uniform(0.0, 1.0)]), unit_likelihood)
        init_kwargs = dict(n_dim=model.ndim)
    else:
        raise ValueError(f"No nested sampler called '{nested_sampler}'.")

    if nested_sampler != 'nestle':
        init_keys = inspect.signature(sampler_init).parameters.keys()
        sig = inspect.signature(sampler_init).bind_partial()
        sig.apply_defaults()
        for key in kwargs.keys() & init_kwargs.keys():
            warnings.warn(f"Value of key '{key}' overwritten.")
        init_kwargs = {
            **{key: kwargs[key] for key in init_keys & kwargs.keys()},
            **init_kwargs}
        sampler = sampler_init(*init_args, **init_kwargs)

    # Run the sampler.
    if nested_sampler == 'nautilus':
        sampler_run = sampler.run
        run_args = ()
        run_kwargs = dict(n_eff=nested_neff, verbose=verbose)
    elif nested_sampler == 'ultranest':
        sampler_run = sampler.run
        run_args = ()
        run_kwargs = dict(
            min_ess=nested_neff, min_num_live_points=nested_nlive,
            show_status=verbose)
    elif nested_sampler == 'dynesty':
        sampler_run = sampler.run_nested
        run_args = ()
        run_kwargs = dict(n_effective=nested_neff, print_progress=verbose)
    elif nested_sampler == 'nestle':
        sampler_run = nestle.sample
        run_args = (likelihood_function, model.prior_transform, model.ndim)
        run_kwargs = dict()
    elif nested_sampler == 'pocomc':
        sampler_run = sampler.run
        run_args = ()
        run_kwargs = dict(n_total=nested_neff, progress=verbose)

    run_keys = inspect.signature(sampler_run).parameters.keys()
    sig = inspect.signature(sampler_run).bind_partial()
    sig.apply_defaults()
    for key in kwargs.keys() & run_kwargs.keys():
        warnings.warn(f"Value of key '{key}' overwritten.")
    run_kwargs = {
        **{key: kwargs[key] for key in run_keys & kwargs.keys()},
        **run_kwargs}

    run_return = sampler_run(*run_args, **run_kwargs)
    #for key in kwargs.keys() - (init_kwargs.keys() | run_kwargs.keys()):
    #    warnings.warn(f"Key '{key}' not recognized by the sampler.")

    # Extract samples, weights, and likelihoods.
    if nested_sampler == 'nautilus':
        obj = sampler
        points, log_w, log_like = sampler.posterior()
        if nautilus_transform_in_likelihood:
            # nautilus applied the identity, so map the unit cube points to
            # parameter space here, one at a time as nautilus does it
            points = np.array([model.prior_transform(u) for u in points])
        log_z = sampler.log_z
    elif nested_sampler == 'ultranest':
        obj = run_return
        points = np.array(run_return['weighted_samples']['points'])
        log_w = np.log(np.array(run_return['weighted_samples']['weights']))
        log_like = np.array(run_return['weighted_samples']['logl'])
        log_z = run_return['logz']
    elif nested_sampler == 'dynesty':
        obj = sampler
        points = sampler.results["samples"]
        log_w = sampler.results["logwt"]
        log_like = sampler.results["logl"]
        try:
            log_z = sampler.results["logz"][-1]
        except:
            log_z = None
    elif nested_sampler == 'nestle':
        obj = run_return
        points = run_return["samples"]
        log_w = run_return["logwt"]
        log_like = run_return["logl"]
        log_z = None
    elif nested_sampler == 'pocomc':
        obj = sampler
        units, log_w, log_like, _ = sampler.posterior(return_logw=True)
        points = np.array([model.prior_transform(u) for u in units])
        log_z = sampler.evidence()[0]

    dur = time.time() - go

    return dict(points=points, log_weight=log_w, log_like=log_like,
                duration=dur), obj

import random, math, inspect

def _invoke_picker(f, rng):
    # Call f() if it takes 0 args, or f(rng) if it takes 1 arg
    try:
        params = [p for p in inspect.signature(f).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                  and p.default is p.empty]
        arity = len(params)
    except (ValueError, TypeError):
        arity = 0
    if arity == 0:
        return f()
    elif arity == 1:
        return f(rng)
    else:
        raise TypeError(f"Sampler callable must take 0 or 1 arg, got {arity}")

def build_random_param_maps(estimator, spec, n_samples=50, seed=42, dedupe=True):
    """
    spec: {param_name: sampler}
      sampler can be:
        - ("choice", [v1, v2, ...])
        - ("uniform", low, high)
        - ("loguniform", low, high)   # low>0
        - ("int_uniform", low, high)  # inclusive
        - callable()                  # or callable(rng)
    """
    rng = random.Random(seed)

    # zero-arg closures that capture rng internally
    def _choice(opts):      return (lambda: rng.choice(opts))
    def _uniform(a, b):     return (lambda: rng.uniform(a, b))
    def _loguniform(a, b):  return (lambda: math.exp(rng.uniform(math.log(a), math.log(b))))
    def _int_uniform(a, b): return (lambda: rng.randint(a, b))

    factories = {
        "choice": _choice,
        "uniform": _uniform,
        "loguniform": _loguniform,
        "int_uniform": _int_uniform,
    }

    pickers = {}
    for name, cfg in spec.items():
        if callable(cfg):
            pickers[name] = cfg            # could be 0-arg or 1-arg
        else:
            kind, *args = cfg
            pickers[name] = factories[kind](*args)  # always 0-arg

    maps, seen = [], set()
    attempts, max_attempts = 0, max(n_samples * 5, n_samples + 10)

    while len(maps) < n_samples and attempts < max_attempts:
        attempts += 1
        sampled = {k: _invoke_picker(pickers[k], rng) for k in pickers}
        pm = {estimator.getParam(k): v for k, v in sampled.items()}
        if dedupe:
            key = tuple(sorted(sampled.items()))
            if key in seen:
                continue
            seen.add(key)
        maps.append(pm)

    if not maps:
        raise RuntimeError("No parameter maps generated (likely all duplicates). "
                           "Try n_samples up, widen ranges, or dedupe=False.")
    return maps
"""
USAGE:

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
gbt = GBTClassifier(featuresCol="features", labelCol="label")


# Build random maps against the *gbt stage*:
spec = {
    "maxDepth": ("int_uniform", 3, 12),
    "maxBins":  ("int_uniform", 32, 256),
    "stepSize": ("loguniform", 1e-3, 2e-1),
    "subsamplingRate": ("uniform", 0.6, 1.0),
    "maxIter":  ("int_uniform", 30, 200),
}

# where gbt is the 
gbt_param_maps = build_random_param_maps(gbt, spec, n_samples=80, seed=7)

cv = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=gbt_param_maps,
    numFolds=3,
    seed=7,
)





"""
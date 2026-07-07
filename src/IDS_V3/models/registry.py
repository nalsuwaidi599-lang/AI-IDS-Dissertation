# keeps track of available models so we can add new ones easily

MODELS = {}

def register(name, builder_fn):
    MODELS[name] = builder_fn

def get(name, **kwargs):
    if name not in MODELS:
        raise ValueError(f"no model called '{name}', available: {list(MODELS.keys())}")
    return MODELS[name](**kwargs)

def available():
    return list(MODELS.keys())

import importlib


_BACKENDS = {}


def register_backend(name):
    def register(factory):
        if name in _BACKENDS:
            raise ValueError(f'Backend already registered: {name}')
        _BACKENDS[name] = factory
        return factory
    return register


def create_backend(name, query):
    if ':' in name:
        module, symbol = name.split(':', 1)
        factory = getattr(importlib.import_module(module), symbol)
    else:
        if name not in _BACKENDS:
            raise ValueError(f'Unknown backend {name}; registered: {sorted(_BACKENDS)}')
        factory = _BACKENDS[name]
    backend = factory(query)
    if not all(callable(getattr(backend, method, None)) for method in ('caption', 'embed', 'ground')):
        raise TypeError('A grounding backend must implement caption, embed, and ground.')
    return backend

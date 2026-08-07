__version__ = "0.0.2"
#from .core import deconverse
def __getattr__(name):
    if name == "deconverse":
        from .core import deconverse
        return deconverse
    raise AttributeError(name)

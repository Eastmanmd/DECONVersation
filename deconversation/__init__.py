__version__ = "0.1.0"
#from .core import deconverse
def __getattr__(name):
    if name == "deconverse":
        from .core import deconverse
        return deconverse
    raise AttributeError(name)

"""

Implentation of helper functions for loading attention models or other dynamically
specified scripts, and for parsing configurations to be passed to those scripts.

An attention model specifically is a python file defining a function `attn_model`
in its globals, that takes any number of configuration arguments passed through
`load_model`'s keyword arguments, and returns a dictionary of `LayerMod`'s that 
should be applied.

### Functions
- `load_model` : Load an attention model.
- `load_cfg` : Load configs from JSON or a key-value string format.

"""



import importlib.util
import json



def load_model(filename, **kwargs):
    """
    Load the global variable `model` from filename, the implication being that
    the object is a `network_manager.LayerMod` to use as an attention model.
    """
    # Run the model file
    fname_hash = hash(filename)
    spec = importlib.util.spec_from_file_location(
        f"model_file_{fname_hash}", filename)
    model_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_file)
    return model_file.attn_model(**kwargs)


def load_cfg(string):
    """
    Either load the JSON file pointed to by `string` or parse it as configs
    (`:`-sepearated statements of KEY=VALUE, where `VALUE` is parsed as a
    Python expression or string if evaluation fails, for example
    `"layer=(0,1,0):beta=4.0"` would become {'layer':(0,4,0), 'beta': 4.0})
    """
    if string is None:
        return {}
    elif string.endswith('.json'):
        with open(string) as f:
            return json.load(f)
    else:
        def try_eval(s):
            try: return eval(s)
            except: return s
        return {
                try_eval(s.split('=')[0].strip()): # LHS = key
                try_eval(s.split('=')[1].strip())  # RHS = val
            for s in string.split(':')
        }







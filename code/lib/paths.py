import os

class _Directory:
    def __init__(self, path):
        self.path = path
    def join(self, *args, **kwargs):
        return os.path.join(self.path, *args, **kwargs)
    def __call__(self, *args, **kwargs):
        return os.path.join(self.path, *args, **kwargs)
    def __add__(self, other):
        return os.path.join(self.path, other)
    def __str__(self):
        return self.path
    

code  = _Directory(os.environ['CODE'])
data  = _Directory(os.environ['DATA'])

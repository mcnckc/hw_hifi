from torch.cuda.amp import autocast
from contextlib import contextmanager


@contextmanager
def optional_autocast(enabled=True):
    if enabled:
        with autocast():
            yield
    else:
        yield

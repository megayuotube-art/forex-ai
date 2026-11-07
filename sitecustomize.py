import pathlib

old_read_text = pathlib.Path.read_text
def new_read_text(self, *args, **kwargs):
    try:
        return old_read_text(self, *args, **kwargs)
    except (NotADirectoryError, FileNotFoundError):
        if any(name in str(self) for name in ['estimator.css', 'params.css', 'features.css']):
            return ""
        raise

pathlib.Path.read_text = new_read_text
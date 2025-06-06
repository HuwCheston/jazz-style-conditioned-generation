# CLaMP-3

This folder contains a modified version of the original [CLaMP-3 repository](https://github.com/sanderwood/clamp3).

In particular, we have added a number of `__init__.py` files necessary to import the functions contained within this
module inside `./jazz_style_conditioned_generation/*.py` files. We have also modified line 66 in `config.py` to use the
`C2` version of the model, required for processing symbolic music (
see [this section of the README.md on the repository](https://github.com/sanderwood/clamp3?tab=readme-ov-file#how-to-switch-between-versions))

To install CLaMP-3 inside the project repository, run the following lines from the root directory (after cloning the
repo and installing the main `requirements.txt`):

```
pip install -r clamp3/requirements.txt
pip install clamp3
```
# Style-Conditioned Symbolic Jazz Generation
[![Tests](https://github.com/HuwCheston/jazz-style-conditioned-generation/actions/workflows/tests.yml/badge.svg)](https://github.com/HuwCheston/jazz-style-conditioned-generation/actions/workflows/tests.yml)

This repository accompanies our paper "Performer and Subgenre Conditioned Generation of Jazz Piano Music". For more
information,check out
the [interactive web application.](https://huwcheston.github.io/jazz-style-conditioned-generation/index.html)

The code in this repository was developed and tested using the following configuration:

- Ubuntu 22.04.1
- Python 3.10.12
- CUDA 12.2

Full Python dependencies can be found inside the [
`requirements.txt` file](https://github.com/HuwCheston/deep-pianist-identification/blob/main/requirements.txt).

## Contents:

- [Installation](#installation)
- [Inference](#inference)
- [Tests](#tests)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation

### Setup

Run the following lines to clone the repository and install the project requirements:

```
git clone https://github.com/HuwCheston/jazz-style-conditioned-generation.git
cd jazz-style-conditioned-generation
python3 -m venv venv     # not necessary, but good practice
source venv/bin/activate
pip install -r requirements.txt
pip install .
```

You now need to install `clamp3`. To make this easier, inside this repo we include a version of the `clamp3` repository
cloned from the [original repo](https://github.com/sanderwood/clamp3), with a few changes necessary to allow it to work
with our filestructure. For more details as to the changes we have made, see `clamp3/README.md` inside this repository.

First, you'll need to download the `clamp3` checkpoint for symbolic music. The file you need can be downloaded via
HuggingFace
at [this link](https://huggingface.co/sander-wood/clamp3/blob/main/weights_clamp3_c2_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth).
and should be placed inside `jazz-style-conditioned-generation/clamp3`.

You can then install the necessary python dependencies and the `clamp3` module itself:

```
pip install -r clamp3/requirements.txt
pip install clamp3
```

### Download data and checkpoints

Data and checkpoints are required regardless of if you want to run inference or train from scratch. These can be
downloaded from [our Zenodo archive](https://doi.org/10.5281/zenodo.15610452) as a single `.zip` file. The folder
structure of the `.zip` is identical to this
repository, so if you unzip it to the root directory (`jazz-style-conditioned-generation`), you should end up with
something like the following:

<details>
<summary>View filestructure</summary>

```
.
└── jazz-style-conditioned-generation/
    ├── checkpoints/
    │   ├── finetuning-customtok-plateau/    # pretrained on ATEPP
    │   │   └── finetuning_customtok_10msmin***/
    │   │       ├── tokenizer.json    # dumped tokenizer configuration  
    │   │       └── validation_best.pth    # pytorch model
    │   ├── pretraining-custom-tokenizer-fixed-preprocessing    # finetuned on jazz
    │   │   └── pretraining_customtok_10msmin***/
    │   └── reinforcement-customtok-plateau    # finetuned + DPO-P
    ├── config/    # same file structure as `checkpoints/`
    │   ├── finetuning-customtok-plateau/    # individual folders per experiment
    │   │   └── finetuning_customtok_10msmin***.yaml    # individual .yaml files per run
    │   ├── pretraining-custom-tokenizer-fixed-preprocessing
    │   │   └── ***.yaml 
    │   └── reinforcement-customtok-plateau
    │   │   └── ***.yaml 
    ├── data/
    │   ├── pretraining/
    │   │   └── atepp/
    │   │       ├── one_folder_per_recording/
    │   │       └── ...
    │   └── raw/    # one folder per source dataset
    │       ├── bushgrafts
    │       ├── jja
    │       ├── jtd/
    │       │   ├── one_folder_per_recording/    # all recordings have the same directory structure
    │       │   │   ├── piano_midi.mid    # raw MIDI data
    │       │   │   └── metadata_tivo.json    # scraped TiVo metadata
    │       │   └── ...
    │       ├── pijama
    │       └── pianist8
    └── README.md     # --> you are here!
```

</details>

The time typically required to install the repository and all dependencies on a "normal" desktop computer is under 20
minutes.

## Inference

To run inference with a model, you can run the following script.

```
python jazz_style_conditioned_generation/generate.py --config <path-to-yaml>
```

To use the fine-tuned checkpoint we provide, you'd run

```
python jazz_style_conditioned_generation/generate.py --config finetuning-customtok-plateau/finetuning_customtok_10msmin_lineartime_moreaugment_init6e5reduce10patience5_batch4_1024seq_12l8h768d3072ff.yaml
```

We provide a few additional flags that can be used to control aspects of the generation:

<details>
<summary>View custom flags</summary>

- `--primer-midi`/`-m`: set this to point towards a custom `.mid` file to use as a primer in generation.
- `--n-primer-tokens`/`-n`: set this to control the number of tokens that will be used as a prompt to the model before
  beginning generation.
- `--sequence-len`/`-l`: set this to control the total length of the sequence to generate: must be longer than the
  number of primer tokens.
- `--top-p`/`-p`: control the value of `p` to use in nucleus sampling.
- `--temperature`/`-t`: control the temperature value to use when scaling the probability distribution
- `--pianist`: use this to pass custom pianists to condition the generation. Multiple values are accepted, e.g. `
  --pianist 'Brad Mehldau' --pianist 'Keith Jarrett' will generate MIDI conditioned on Mehldau and Jarrett.
- `--genre`: use this to pass custom genres to condition the generation. Again, multiple values are accepted, e.g. ``
  -genre 'Straight-Ahead Jazz' --genre 'Traditional & Early Jazz'`
- `--time-signature`: custom time signature to use in conditioning. Only one value is accepted, and must be either `3`
  or `4`
- ``--tempo``: custom tempo to use in conditioning, in number of beats per minute. Only one value is accepted, e.g.
  `--tempo 260`
- `--use-track-tokens`: when no custom `--genre`, `--pianist` (etc.) tokens are provided, when `True` we will use the
  metadata already assigned to a track. When `False`, we will not use any conditioning tokens.

</details>

The condition tokens that can be used during inference are found inside
`jazz_style_conditioned_generation/data/conditions.py`.

By default, generation will occur using a random primer MIDI file from the held out dataset. 128 primer tokens will be
used, and generation will occur until the sequence lasts for 1024 tokens or a `End` token is sent by the model,
whichever occurs first. Temperature and top-p sampling will not be used. The generation will inherit the same condition
tokens as the primer file.

Generated examples will be saved in both `.mid` and `.wav` format inside `outputs/generation`. We recommend a GPU with
at least 4GB of VRAM for inference. If no GPU is available, the code will fall back to using CPU, however this will be
very slow.

## Training

We provide three scripts for 1) pre-training the transformer on ATEPP, 2) fine-tuning on the custom jazz dataset, and 3)
running reinforcement learning with DPO-P.

We recommend a GPU with at least 20GB of VRAM for training, using the default batch size and parameters. We used a
single NVIDIA RTX 3090 TI during our training and experiments.

### Pre-training

To run pre-training on ATEPP, ensure that the [data has been downloaded](#installation) and run the following script:

```
python jazz_style_conditioned_generation/training.py --method pretraining --config pretraining-custom-tokenizer-fixed-preprocessing/pretraining_customtok_10msmin_lineartime_moreaugment_linearwarmup10k_1e4_batch4_1024seq_12l8h768d3072ff.yaml
```

During training, training and validation loss will be logged to the console. By default, checkpoints will be dumped
inside `checkpoints/pretraining-custom-tokenizer-fixed-preprocessing/pretraining_customtok_10ms/...`. One checkpoint
will be made after every 10 epochs, and a `validation_best.pth` checkpoint will keep track of the checkpoint with the
best validation loss. The directory to save checkpoints to can be adjusted by setting the `checkpoint_dir` argument
inside the `.yaml` file.

If training is interrupted, it will be resumed from the most recent checkpoint when re-running the above command. Be
aware that if training fails due to running out of memory when saving a checkpoint, you will first need to delete the
corrupted checkpoint(s) in order to resume training. We will raise a custom error when this occurs.

### Fine-tuning

Fine-tuning follows the same general principles as pre-training. To fine-tune on the jazz dataset, ensure that
the [data has been downloaded](#installation) and that you have a pre-training `validation_best.pth` checkpoint (either
created yourself, or downloaded from Zenodo). Then, run the following script:

```
python jazz_style_conditioned_generation/training.py --method finetuning --config finetuning-customtok-plateau/finetuning_customtok_10msmin_lineartime_moreaugment_init6e5reduce10patience5_batch4_1024seq_12l8h768d3072ff.yaml
```

### Reinforcement learning with DPO-P

Reinforcement learning with DPO-P consists of two stages: 1) generating a dataset of examples and 2) learning from
these.

First, we can generate a dataset of N examples for every conditioning token by running

```
python jazz_style_conditioned_generation/reinforcement/rl_generate.py --config <path-to-yaml> --genres 0 1 2 3 4 5 ... --pianists 0 1 2 3 4 5...
```

`genres` should be set according to the number of genres to generate for (20 in total), while pianists should be set the
same (25 in total). `config` should be a path to a `.yaml` file inside `config/reinforcement-customtok-plateau`.

Note that, if you have saved fine-tuning checkpoints to somewhere other than the `checkpoints` folder, you will need to
update the `policy_model_checkpoint` and `reference_model_checkpoint` paths inside the `.yaml` file.

Then, we can run the reinforcement learning script:

```
python jazz_style_conditioned_generation/reinforcement/rl_train.py --config <path-to-yaml>
```

The `--config` flag should be set using the same `.yaml` file as was used when generating. Once this has finished, a
`reinforcement_iteration_k.pth` file will be dumped inside `checkpoints`, where `k` is the number of iterations (
starting from 0).

You can then re-run the `rl_generate.py` script using this checkpoint to create a new dataset of examples for subsequent
tuning with DPO-P.

## Tests

To run all the tests, follow the steps above for [installation](#installation). Then, you can run the following from the
root directory of the repository:

```
python -m unittest discover -s tests
```

By default, we run a few tests that iterate over the entire dataset. These can take quite a while to complete. To skip
these tests, you can set the environment variable `REMOTE` beforehand, like so:

```
REMOTE=true python -m unittest discover -s tests
```

All the tests should pass!

## Citation

TODO

## License

Our work is licensed under a [CC-BY-NC-SA-4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/). Under the
terms of this license, any performances created using our model *cannot* be used for commercial purposes.

## Acknowledgements

We would like to acknowledge the creators and maintainers of the open source datasets used in this project, namely
the [PiJAMA](https://github.com/almostimplemented/PiJAMA), [ATEPP](https://github.com/tangjjbetsy/ATEPP),
and [Pianist8](https://zenodo.org/records/5089279) datasets. We also want to acknowledge the musicians in these
datasets, whose artistic output made this research possible.

This work was completed during the course of a PhD that received funding from
the [Cambridge Trust](https://www.cambridgetrust.org/).
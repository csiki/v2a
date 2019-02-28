# Autoencoded sensory substitution

Visual-to-auditory (V2A) sensory substitution stands for the translation of images to sound,
in the interest of aiding the blind. The generated soundscapes should convey visual information,
ideally representing all the details on the given image, with as short a sound sequences as possible.
Traditional V2A conversion methods apply an explicitly predefined function that transforms the input
image pixel-by-pixel to soundscapes, superimposing them in the final step. Here is the implementation
of a novel conversion approach, which posits sensory substitution as a compression problem.

Optimal compression is learnt and computed by a recurrent variational autoencoder, called AEV2A.
The autoencoder takes an image as input, translates it to a sequence of soundscapes, before
reconstructing the image in an iterative manner drawing on a canvas. The neural network implementation
is based on the [DRAW](https://arxiv.org/abs/1502.04623) model; the repository from which the code was
initially cloned can be found [here](https://github.com/kvfrans/draw-color). AEV2A further builds on
[WaveGAN](https://arxiv.org/abs/1802.04208) (repo [here](https://github.com/chrisdonahue/wavegan)).

For further details check this [blog post](TODO) or the thesis [here](TODO).

![](https://i.imgur.com/ecdDO4s.png) ![](https://i.imgur.com/ALHRqWu.png) ![](https://media.giphy.com/media/2tKDQQuaHNvRGfYQm5/giphy.gif) ![](https://i.imgur.com/11CYllL.png) ![](https://media.giphy.com/media/1wXgFezvfCkjCo2aUh/giphy.gif)

## Requirements
Implementation has been tested on Linux (Ubuntu), but the core learning parts should run regardless of the operating system.
The following instructions are for Ubuntu only. Every library that's in brackets are not necessary for training purposes,
but needed for either dataset generation or testing/visualization of results.

Dataset generation may involve running a Matlab script, if only the contour of the images are planned to fed to the
network. The contour detection Matlab scripts are under `matlab/faster_corf/`, originally cloned from
the [Push-Pull CORF repository](https://www.mathworks.com/matlabcentral/fileexchange/47685-contour-detection-with-the-push-pull-corf-model).
For further information [consult the paper](https://link.springer.com/article/10.1007/s00422-012-0486-6).

- `python >=3.6`, \[`ffmpeg`, `opencv`\]
- Python packages: `numpy`, `scikit-image`, `matplotlib`, `tables`, \[`csv`, `simpleaudio`, `scipy`, `scikit-learn`, `pymatlab`\]
- [`Tensorflow 1.9.0`](https://www.tensorflow.org/install): other versions may work too; GPU package recommended.
```bash
sudo apt-get install python3.6
sudo python3.6 -m pip install
sudo pip3 install numpy scikit-image tables matplotlib
sudo pip3 install tensorflow-gpu  # requires CUDA-enabled GPU
```
```bash
# not mandatory for training purposes:
sudo apt-get install python3-opencv ffmpeg
sudo pip3 install csv simpleaudio scipy scikit-learn pymatlab
```


## How to run
AEV2A is trained unsupervised, meaning, the image set is the only data needed to train the network.
You can either download the hand gesture or table dataset used in the study, or generate your own
from a set of images or videos. Consult the readme under `data/` for further info.

Before starting the training, a configuration has to be defined that contains the hyperparameters of the autoencoder.
All configurations should be specified in the `configurations.json` file. You could just use the `default`
configuration already present in the json file, or create a new one according to the default. In `config.py`
you may find a short description for each parameter; for further detail check the [thesis](TODO).

To start the training process, just run the `aev2a.py` script like so:
```bash
python3.6 aev2a.py config_name data/dataset.hdf5 train
```
You may substitute the name of the configuration and the dataset path, and could add further parameters
as (in order): `number of training epochs`, `frequency of logging` (number of trained batches between logging),
`frequency of saving the model` and a `postfix string for the model` in case there are multiple models with
the same configuration. All of the command line arguments are optional, so for testing purposes you could just
simply run the script like: `python3.6 aev2a.py`.


[TOOD high lvl: test network, run proto, analyze image-to-sound]

## What's what
[TODO very short descr of each folder/file]

## Datasets
[TODO short description, more under /data]

[TODO image of hands dataset]

## Training

### Tensorboard analysis

## Testing

## Running live

## Image-to-sound conversion analysis

## Citation
```
[TODO bibtex]
```

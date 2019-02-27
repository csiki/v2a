#Autoencoded sensory substitution

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

[TODO ADD GIF HERE]

##Requirements
[TODO PYTHON LIBS, FFMPEG, PYMATLAB]

##How to run
[TOOD high lvl: gen dataset/download, train network, test network, run proto, analyze image-to-sound]

##What's what
[TODO very short descr of each folder/file]

##Datasets
[TODO short description, more under /data]

[TODO image of hands dataset]

##Training

###Tensorboard analysis

##Testing

##Running live

##Image-to-sound conversion analysis

##Citation
```
[TODO bibtex]
```

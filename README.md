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

Videos on the visual-auditory correspondence of two models have been compiled and merged. Here are
the videos for the [hand dataset](https://youtu.be/EGPz4HIFsCM)
and [table dataset](https://youtu.be/4IMlWaVQ2fk) trained models.
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
- [`Tensorflow 1.9.x`](https://www.tensorflow.org/install): other versions may work too; GPU package recommended.
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

### Dataset
AEV2A is trained unsupervised, meaning, the image set is the only data needed to train the network.
You can either download the hand gesture or table dataset used in the study, or generate your own
from a set of images or videos. Consult the readme under [`data/`](data/README.md) for further info.

![](https://i.imgur.com/8R9gd9F.png) ![](https://i.imgur.com/gyJX60s.png) ![](https://i.imgur.com/O5obnlu.png) ![](https://i.imgur.com/c2NUw9N.png)

The hand gesture dataset includes contour images of 15 different postures, in varying horizontal and vertical positions.
The table image set depicts contours of either a beer can or a gear on a surface, again in varying positions.

### Training
Before starting the training, a configuration has to be defined that contains the hyperparameters of the autoencoder.
All configurations should be specified in the `configurations.json` file. You could just use the `default`
configuration already present in the json file, or create a new one according to the default. In `config.py`
you may find a short description for each parameter; for further details check the [thesis](TODO).
The default config contains the same parameters as we used in the study to learn hand posture images.

To start the training process, just run the `aev2a.py` script like so:
```bash
python aev2a.py config_name data/dataset.hdf5 train
```
You may substitute the name of the configuration and the dataset path, and could add further parameters
as (in order): `number of training epochs`, `frequency of logging` (number of trained batches between logging),
`frequency of saving the model` and a `postfix string for the model` in case there are multiple models with
the same configuration. All of the command line arguments are optional, so for testing purposes you could just
simply run the script like: `python aev2a.py`.

### Tensorboard analysis
If you run the training with a `frequency of logging` greater than zero (100 by default), tensorboard summaries
are produced and stored under `summary/`. Summaries include training and test loss plots,
sound property (frequency, amplitude, source location) distributions, drawn image and synthesized audio.
[Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard) provides an easy, fast way
to assess the efficacy of your model, run it like:

```bash
tensorboard --logdir=./summary  # if you are in the root folder of v2a
# now open a web browser and type the url 'localhost:6006' to open the dashboard
```

The long model names, defined by the model parameters, come handy in tensorboard, where one can filter
trained models using regex for the purpose of comparing them.

### Video generation
If you installed the optional packages above, you can generate videos that play the soundscapes alongside
the drawing, as a mean for intuitively assessing the sound-to-visual correspondence.
Videos are stored under the `vids/` folder, one video for each input image of one batch
and one extra that is just the concatenation of the rest.
 
```bash
python aev2a.py config_name data/dataset.hdf5 test 0 0 model_name_postfix
```

If you have only one model for the given configuration set, you may leave out the last three parameters.

### Demoing the model on images
The `test_on_imgs.py` script initiates a selected trained model, feeds images to it from the given dataset
and plays the soundscape synthesized from the image. You can change images front and back by pressing `D` and `A` keys.
You can select whether the sequence of images are chosen randomly or are predetermined. It may take the input images
from either the train or the test set.

```bash
python test_model.py cfg_name test seq model_name_postfix
```

This script could be used as an experimental tool, in which the presented image has to be named by the listener, and
thus, the discrimination accuracy can be assessed.

### Live demo
You may run your AEV2A model live by taking video with your Android phone and listening to the corresponding
audio representation at the same time. In our implementation, the captured video is streamed to your PC
where it gets translated into sound so you may listen to it. Ideally, you would place your phone
inside a VR helmet/cardbox, fastening the camera at eye level; headphones are essential.

`run_proto.py` runs the live demo. Similarly to the dataset generation phase, you can set whether to
apply the more sophisticated CORF edge detection algorithm (Matlab required) or just Sobel.

To set up your Android phone with the trained AEV2A model, you first need to:
0. Install the IP Webcam app from Google Play, launch it and set the video resolution to `320x240` under Video preferences.
1. Connect your phone via USB to the computer that runs the script
2. Turn on USB tethering on the phone, but turn off WiFi and mobile data
3. Launch the IP Webcam app and press "Start server".
4. Start the `run_proto.py` script with parameters providing whether to run in "test" or "fast" mode
(test mode shows how the contour image and the decoder reconstructed image looks like real time),
the edge detection algo to apply, the mobile ip of your phone (displayed in the IP Webcam app),
the name of the model configuration and postfix if used any.

```bash
python run_proto.py test corf mobile_ip config_name model_name_postfix
```

After the model is loaded, you should be seeing three windows of images showing the original, contour and
reconstruction stages. The audio should be playing at the same time, too.

### Image-to-sound conversion analysis

[TODO]

### What else
run audio_gen separately to test different soundscapes
[TODO very short descr of each folder/file]

## Model structure
_x_ is the input image, _c<sub>t</sub>_ is the state of the canvas at iteration _t_.
_h<sub>t</sub>_ is the hidden state of either the encoder or the decoder RNN (LSTM).
_z<sub>t</sub>_ is drawn from a Normal distribution parametrized by the output of the
encoder network. _a<sub>t</sub>_ is the audio representation, that is, series of
frequency, amplitude and spatial modulated soundstreams.

![](https://i.imgur.com/Q0YAAna.png)

## Citation
```
[TODO bibtex]
```

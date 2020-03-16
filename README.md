# Moments
A Python application which captures moments of happiness and surprise from a video stream of one or more people using Computer Vision and Deep Learning. GUI designed using Qt 4 Designer.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

This project needs python 2 installed on the local machine along with *cv2*, *numpy*, *statistics* and *keras* which can be installed using:

```bash
$ python -m pip install cv2
$ python -m pip install numpy
$ python -m pip install keras
$ python -m pip install statistics
```


## Running

main.py defines all the functions and can be run using:

```bash
$ python main.py
```

This will run a GUI where the video stream and capturing can be controlled using **Start** and **Stop**

***TODO:*** Automatic capturing using countdown timer

## Deep Learning Model

The model used is from this [research paper](https://github.com/oarriaga/face_classification/blob/master/report.pdf) written by Octavio Arriaga, Paul G. Plöger, and Matias Valdenegro.

![Model](https://i.imgur.com/vr9yDaF.png?1)

## Credit

* Computer vision powered by OpenCV.
* [Github Link](https://github.com/petercunha/Emotion/) containing the code of the pretrained model.
* Neural network scaffolding powered by Keras with Tensorflow.
* Convolutional Neural Network (CNN) deep learning architecture is from this [research paper](https://github.com/oarriaga/face_classification/blob/master/report.pdf).
* Pretrained Keras model and much of the OpenCV code provided by GitHub user [oarriaga](https://github.com/oarriaga).

# Speech Emotion Decoder Project

The dataset is a collection of utterances extracted recorded in a controlled environment. This dataset is rich and diverse, featuring utterances from multiple speakers.

Key characteristics of the dataset:

* Utterances: The dataset comprises various utterances from different actors. The sampling
rate of the audios is 44100.

* Emotional valence: Each utterance in the dataset has annotations for emotional valence,
a float number between 1 and 5.

Goal: build a Deep Learning model that is capable of estimating the emotional valence
of a recording. 

## Running Code:
*main_notebook.ipynb* contains all required code to run in jupyter notebook style

To run code in a traditional python file format run *main.py*

## Results:

- Found that the best Optimizer is Adagrad with 13 epochs 

- The 1D CNN model performs best with the hidden sizes of XXX and a learning rate of XXX

- Achieved MSE of 0.51485 on the test data. Placing 7th out of 52 teams in school competition

- Achieved a 9.4 out of 10 on the project

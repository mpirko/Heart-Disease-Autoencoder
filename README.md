# Heart Disease Autoencoder

This code is an implementation of an autoencoder using Keras and TensorFlow libraries. An autoencoder is an unsupervised machine learning technique used to learn a compressed representation of the input data. The model is trained to learn a compressed representation of the input data and then generate output that is as close to the input as possible. This code is applied to a heart disease dataset.

The code reads the dataset using Pandas and preprocesses it. Then, it splits the data into train and test sets and applies normalization using the QuantileTransformer function from the Scikit-learn library. The code then builds and trains the autoencoder using Keras with hyperparameter tuning using the Keras Tuner library. The autoencoder's performance is evaluated using the mean squared error and visualizing the reconstructed data. The code plots the loss and reconstruction of the validation dataset, and lastly, the code evaluates the model on the test set.

The implications of using autoencoders are vast. This technique can be used to detect anomalies and outliers in datasets, compress images, and generate new images. In the medical field, autoencoders can be used for disease diagnosis and treatment. However, as with any machine learning model, the data quality and the model's tuning are critical factors in the model's performance.
Getting Started

### To get started with this project, you will need to install the following libraries:

keras-tuner
pandas
numpy
sklearn
matplotlib
tensorflow

### The code generates 5 different models:

NoHyperparameter.h5
Hyperparameter.h5
EarlyStopHyperparameter.h5
ExtraHiddenLayer.h5
OneLessHiddenLayer.h5

Once you have these libraries installed, you can run the code in any of the 5.py files. The code reads the dataset using Pandas and preprocesses it. Then, it splits the data into train and test sets and applies normalization using the QuantileTransformer function from the Scikit-learn library. The code then builds and trains the autoencoder using Keras with hyperparameter tuning using the Keras Tuner library. The autoencoder's performance is evaluated using the mean squared error and visualizing the reconstructed data. The code plots the loss and reconstruction of the validation dataset, and lastly, the code evaluates the model on the test set.


## Implications

The implications of using autoencoders are vast. This technique can be used to detect anomalies and outliers in datasets, compress images, and generate new images. In the medical field, autoencoders can be used for disease diagnosis and treatment. However, as with any machine learning model, the data quality and the model's tuning are critical factors in the model's performance.

## Usage

To use this code, follow these steps:

Clone this repository to your local machine.
Install the required libraries.
Open the .py file and run the code.
The models will be saved in the current working directory with the following filenames:
NoHyperparameter.h5
Hyperparameter.h5
EarlyStopHyperparameter.h5
ExtraHiddenLayer.h5
OneLessHiddenLayer.h5
Use the saved models as needed for your specific use case.

The base model I used had 5 neurons in the latent space, 2 hidden layers with 32 neurons and 16 neurons in them, resepctively, and no hyperparameter tuning. I added features such as hyperparameter tuning, an early stop, an extra hidden layer (at 64 neurons) and removed

For this dataset, I found that the regular Hyperparameter model produced the best results in terms of FMSE. I found this interesting because showing the training and validation loss over time, it seemed to drastically differ over time as opposed to the early stop model, where it stopped when these two values were more closely aligned. This suggests that even though the training and validation loss can have a greater validation loss over time, it appears that the FMSE decreases over time. I would hypothesize this is due to the model memorizing the values rather than true learning. I also find it interesting how the one less and one more hidden layer autoencoders with early stop had the same FMSE values, while the 2 neuron early stop model had a higher value. I would like to look into how the neuron number affects the FMSE in the future. 

<br>
<img width="493" alt="Screen Shot 2023-05-15 at 9 21 21 AM" src="https://github.com/mpirko/Heart-Disease-Autoencoder/assets/69722618/f2c22e00-460c-4909-bb31-705c1d3cb4c6">
<br>

You can see my work in visualizing the training loss and validation loss over each epoch in the .ipynb files. You can also see the FMSE per feature in each file and see the feature importances for each value. All 5 models depicted in descending order of feature importance: Major Vessels colored by Fluoroscopy, then Chest Pain Type, then Resting Blood Pressure, then Thalassemia, then ST depression by exercise, then Sex, then ST Depression Induced by Exercise Relative to Rest, then RestingECG, then cholesterol, then ExerciseAngina, then Fasting Blood Sugar, then Age (Years), then Maximum Heart Rate.

<br>
<img width="493" alt="Screen Shot 2023-05-14 at 11 55 07 PM" src="https://github.com/mpirko/Heart-Disease-Autoencoder/assets/69722618/09beaa82-7791-461d-8a20-0edb64958559">
<br>

In the future I hope to modify the code to add two different latent spaces as there are two types of data distributions present: normal and discrete, so I would want to optimize each of the different types individually to get the best results for this data set.
<br>
<img width="952" alt="Screen Shot 2023-05-14 at 11 59 08 PM" src="https://github.com/mpirko/Heart-Disease-Autoencoder/assets/69722618/16d35f0c-77d7-4e25-aa11-7d721201a69f">
<br>

## Acknowledgments


This project is based on the following resources:

Keras documentation
TensorFlow documentation
Keras Tuner documentation
Scikit-learn documentation
Heart Disease Dataset from UCI Machine Learning Repository

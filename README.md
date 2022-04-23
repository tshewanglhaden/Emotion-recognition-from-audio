
# Speech Emotion Detection using machine learning

Primary mode of communication between human beings is speech which is defined as the
expression of thoughts and feelings mainly to convey ideas and information in a speaker’s mind.
This information is encoded in the speech signal. The production of speech from an individual is
greatly altered by a person’s emotional state. The resulting speech signal characterizes different
emotions which are of prime importance in analyzing the mental state of an individual.
Using the speech emotion recognition (SER) tool, different speech signals are processed and
classified to detect different emotions in a human being using a machine learning model


## Datasets

Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) used which can be found on https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

I used audio files to detect emotions from RAVDESS dataset which contains 1440 audio files of 24 actors, male and female. 
Speech includes calm, happy, sad, angry, fearful, surprise, and disgust expressions. Each of the RAVDESS files has a unique filename. The filename consists of a 7-part numerical identifier as explained by https://smartlaboratory.org/ravdess/

## Libraries

[![1.jpg](https://i.postimg.cc/yxTy1TQq/1.jpg)](https://postimg.cc/H8Vy3b13)



## Audio signal analysis 
First, one sample audio file is loaded from the dataset and played  using IPython.Display and Python’s librosa library: [![2.jpg](https://i.postimg.cc/X7ptHGpj/2.jpg)](https://postimg.cc/3dQLWwXV)


## Feature extraction
For feature extraction, we need to read every audio file using the **soundfile**.Features are extracted using feature extraction processes. Here,  I used mel frequency cepstral coefficients (mfcc), mel and chroma. **Librosa** library is used which is one of the libraries used for audio analysis.
[![3.jpg](https://i.postimg.cc/mktz1NV1/3.jpg)](https://postimg.cc/LgF6rPcm)

## To detect emotion
After feature extractions, data are stored in the result. After data files are loaded using **glob**, 20% of dataset are taken for testing processes. Four emotions are focused:calm, happy, fearful, disgust.Every emotion in audio files are stored to the emotion and then files are uploaded to the feature extraction. Feature extracted are stored to the x list and emotion of file is stored to the y list. Then data is tested using test_train_split.Audio file shape is converted into array using numpy 
 
[![4.jpg](https://i.postimg.cc/nc8TFzn2/4.jpg)](https://postimg.cc/t75FD9Zn)

## Building model 
Multilayer Perceptron (MLP) from **sklearn** is used as a classifer having alpha rate as 0.01, batch size 256, increased hidden layer as 300 (to get accurate output)and adaptive activation function. Finally x_train data, y_train data are fitted to the model (MLP classifier). After training the model, test set which is given to the prediction are predicted using the **model.predict function**. Output of model.predict is stored to the y_predict. Then, accuracy of the model is calcuated using **accuray score** and printed the accuracy value in terms of percentage.

[![5.jpg](https://i.postimg.cc/wj59f7jG/5.jpg)](https://postimg.cc/4YdCn4t6)

## Predictions
After tuning the model, tested it out by predicting the emotions for the test data.
[![6.jpg](https://i.postimg.cc/VkzNcjp8/6.jpg)](https://postimg.cc/7fQ43Tqs)

## Testing a new audio data
In order to test the model, a new audio data file is tested and it's prediction is observed. For that file, preprocess was again carried out to extract the features from that data itself. Feature of the input data is converted into array format using **[]**. Array format data is given to the mlp model and the output of that is stored to the y_pre and printed the prediction. 
[![7.jpg](https://i.postimg.cc/JnxfZkKb/7.jpg)](https://postimg.cc/wyt4d3Cv)  
The audio contained a female voice which said **"Dogs are sitting by the door"** in a calm tone. RAVDESS unique file identifier for this audia was **03-01-02-02-02-02-10** 

We can observe in the code (ser.ipynb) that the model has also predicted the emotion of the audio as **calm**. 

## References
https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

https://towardsdatascience.com/speech-emotion-recognition-using-ravdess-audio-dataset-ce19d162690




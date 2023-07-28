# README

This repository contains code for building and training several deep learning models for time series classification using TensorFlow as the second homework of Artificial Neural Networks and Deep Learning (AN2DL) course (A.Y 22/23) at Politecnico  di Milano. The models implemented here are as follows:

1. LSTM Classifier
2. Bidirectional LSTM (BiLSTM) Classifier
3. 1D Convolutional Neural Network (1DCNN) Classifier (Version 1)
4. 1D Convolutional Neural Network (1DCNN) Classifier (Version 2)

## Introduction

The goal of this project is to develop models that can accurately classify time series data into one of multiple classes. The dataset used in this project consists of time series sequences and their corresponding labels. The models are evaluated based on various classification metrics, such as accuracy, precision, recall, and F1 score.

## Requirements

Before running the code, make sure you have the following libraries and tools installed:

- TensorFlow (version >= 2.0)
- Numpy
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn

## Data Loading & Preprocessing

The data is loaded from two numpy files: `x_train.npy` and `y_train.npy`, which contain the input sequences and their corresponding labels, respectively. The shapes of `x_train` and `y_train` are printed after loading.

The data is preprocessed using standardization. The `standardize` function is applied to the input data before training the LSTM models. The data is then split into training, validation, and testing sets.

## LSTM Classifier

The LSTM Classifier is implemented using the Keras API in TensorFlow. The model architecture consists of two Bidirectional LSTM layers followed by a Global Max Pooling layer. The output is passed through a dense layer and a final softmax layer to produce class probabilities.

The model is trained for a fixed number of epochs, and callbacks are used to monitor validation loss and reduce the learning rate if the validation loss plateaus.

## Bidirectional LSTM (BiLSTM) Classifier

The BiLSTM Classifier is also implemented using the Keras API in TensorFlow. The architecture includes two Bidirectional GRU (Gated Recurrent Unit) layers followed by a Global Max Pooling layer. The output is passed through dense layers and a final softmax layer for classification.

The model is trained similarly to the LSTM Classifier with the use of callbacks.

## 1D Convolutional Neural Network (1DCNN) Classifier (Version 1)

The 1DCNN Classifier (Version 1) is implemented using the Keras API in TensorFlow. The architecture consists of two 1D Convolutional layers followed by Max Pooling and Dense layers.

The model is trained using the same training setup as the previous models.

## 1D Convolutional Neural Network (1DCNN) Classifier (Version 2)

The 1DCNN Classifier (Version 2) is another variant of the 1DCNN model. The architecture is similar to Version 1 but with some changes in the number of layers and activation functions.

The model is trained using the same training setup and callbacks.

## Model Evaluation

After training, each model's performance is evaluated on the test set. The classification metrics such as accuracy, precision, recall, and F1 score are computed. Additionally, the confusion matrix is generated to analyze the model's performance for each class.

## Note on Unsuccessful Attempts

The README also mentions some unsuccessful attempts to improve model performance using techniques such as Data Augmentation, ResNet architecture, and Multi-Head Attention. However, these attempts were not effective in improving the model's performance, so they were not included in the final implementation.

## Conclusion

This repository provides code for building and training deep learning models for time series classification. The LSTM, BiLSTM, and 1DCNN models are implemented and evaluated on the provided dataset. Users can try different variations of these models or experiment with other architectures to further enhance the model's performance.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions or inquiries related to this project, you can contact [Kiarash Rezaei](mailto:kiarashrezaei@yahoo.com).


Please ensure to install the required libraries mentioned in the "Requirements" section before running the code. Happy experimenting with time series classification!

# FER2013 Dataset - Facial Emotion Recognition using Convolutional Neural Networks

In this project, we have used the FER2013 dataset for facial emotion recognition using Convolutional Neural Networks (CNNs). We have used TensorFlow as the deep learning framework and have implemented the CNN model using Keras API.

## Dataset

The FER2013 dataset contains 35,887 grayscale, 48x48 sized face images with seven emotions - angry, disgusted, fearful, happy, neutral, sad, and surprised. The dataset is divided into training, validation, and testing sets.

## Requirements

* Python 3.x
* TensorFlow 2.x
* NumPy
* Pandas
* Seaborn
* Matplotlib

## Model Architecture

We have implemented a 6-layer CNN model with the following layers:
* Conv2D layer with 32 filters, 3x3 kernel size, and 'relu' activation function.
* Conv2D layer with 64 filters, 3x3 kernel size, and 'relu' activation function.
* BatchNormalization layer
* MaxPooling2D layer with pool size 2x2
* Dropout layer with 25% rate
* Conv2D layer with 128 filters, 3x3 kernel size, 'relu' activation function, and L2 regularization.
* Conv2D layer with 256 filters, 3x3 kernel size, and 'relu' activation function.
* BatchNormalization layer
* MaxPooling2D layer with pool size 2x2
* Dropout layer with 25% rate
* Flatten layer
* Dense layer with 1024 units and 'relu' activation function.
* Dropout layer with 50% rate
* Dense layer with 7 units (number of emotions) and 'softmax' activation function.

## Training

We have used ImageDataGenerator for image augmentation and created training and testing sets. The model is trained for 60 epochs with a batch size of 64 using Adam optimizer with learning rate 0.0001 and decay 1e-6. The model is saved after every epoch with the best validation accuracy using ModelCheckpoint callback. We have also used CSVLogger, TensorBoard, EarlyStopping, and ReduceLROnPlateau callbacks for monitoring and optimizing the training process.

## Results

The model achieved an accuracy of 67% on the testing set.

We have also plotted the model accuracy and loss during training using Seaborn line plots.

We have computed the confusion matrix and classification report for the training set and plotted the confusion matrix as an image.

## Future Work

In the future, we can improve the model performance by increasing the dataset size, fine-tuning the model hyperparameters, and implementing more advanced deep learning techniques.

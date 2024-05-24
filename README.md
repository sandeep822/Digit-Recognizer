# Digit-Recognizer

Dataset Description:

The "Digit Recognizer" competition on Kaggle is designed for individuals with some experience in R or Python and a basic understanding of machine learning, but who are new to computer vision. The challenge centers around the MNIST dataset, which stands for "Modified National Institute of Standards and Technology." This dataset, a cornerstone in computer vision, consists of tens of thousands of handwritten images of digits. Since its release in 1999, MNIST has been instrumental in benchmarking classification algorithms. The competition aims to serve as a hands-on introduction to techniques such as neural networks, offering participants the opportunity to explore and experiment with various algorithms.

The MNIST dataset comprises 42,000 rows and 785 columns. Each row represents an image of a handwritten digit, while the columns correspond to pixel values in a flattened format (28x28 pixels). The first column, labeled "label," indicates the digit (0-9) that the corresponding image represents. The pixel columns (pixel0 to pixel783) contain grayscale values ranging from 0 to 255, representing the intensity of each pixel in the image. This classic dataset remains a valuable resource for researchers and learners alike, providing a foundation for testing and comparing classification algorithms.

Goal:

The primary goal of the "Digit Recognizer" competition is to develop accurate models capable of correctly identifying handwritten digits based on the provided dataset. Participants are encouraged to explore a variety of algorithms, ranging from traditional classification methods like Support Vector Machines (SVM) and K-Nearest Neighbors (KNN) to more advanced techniques like neural networks. The challenge emphasizes the application of computer vision fundamentals, and participants are expected to employ their skills in both model development and evaluation. The overarching objective is to foster learning and experimentation within the realm of computer vision while achieving high accuracy in digit classification. The dataset is made available under a Creative Commons Attribution-Share Alike 3.0 license.

![image](https://github.com/sandeep822/Digit-Recognizer/assets/50867031/457e21db-cde5-4844-8d10-de68c95fca88)

A visual representation of the first instance of each digit (0-9) in the training dataset using a 2x5 grid of subplots. For each digit, a corresponding image is displayed, showcasing the unique characteristics of the handwritten digits. The grayscale images, obtained from the 'train_data' dataframe, are reshaped to a 28x28 pixel format. The resulting grid provides a concise and visually appealing overview of the diverse writing styles associated with each digit. This visualization serves as an initial exploration of the dataset, offering a glimpse into the variability and complexity of handwritten digits. It is a valuable step in understanding the input data before embarking on the development of machine learning models for digit recognition.

machine learning pipeline for 5 models- image classification using various models, including Logistic Regression, Random Forest, K-Nearest Neighbors, a simple Convolutional Neural Network (CNN), and Gradient Boosting.

/usr/local/lib/python3.10/dist-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
Model: Logistic Regression
Accuracy: 0.9216666666666666
Classification Report:
               precision    recall  f1-score   support

           0       0.96      0.97      0.96       816
           1       0.96      0.98      0.97       909
           2       0.92      0.90      0.91       846
           3       0.91      0.88      0.90       937
           4       0.92      0.94      0.93       839
           5       0.86      0.88      0.87       702
           6       0.93      0.96      0.95       785
           7       0.93      0.92      0.93       893
           8       0.90      0.88      0.89       835
           9       0.90      0.90      0.90       838

    accuracy                           0.92      8400
   macro avg       0.92      0.92      0.92      8400
weighted avg       0.92      0.92      0.92      8400

==================================================
Model: Random Forest
Accuracy: 0.9644047619047619
Classification Report:
               precision    recall  f1-score   support

           0       0.98      0.98      0.98       816
           1       0.98      0.99      0.99       909
           2       0.97      0.96      0.96       846
           3       0.95      0.95      0.95       937
           4       0.96      0.97      0.97       839
           5       0.97      0.95      0.96       702
           6       0.96      0.98      0.97       785
           7       0.97      0.96      0.96       893
           8       0.96      0.95      0.95       835
           9       0.94      0.94      0.94       838

    accuracy                           0.96      8400
   macro avg       0.96      0.96      0.96      8400
weighted avg       0.96      0.96      0.96      8400

==================================================
Model: K-Nearest Neighbors
Accuracy: 0.9648809523809524
Classification Report:
               precision    recall  f1-score   support

           0       0.98      1.00      0.99       816
           1       0.94      1.00      0.97       909
           2       0.99      0.94      0.96       846
           3       0.96      0.96      0.96       937
           4       0.98      0.97      0.98       839
           5       0.96      0.95      0.96       702
           6       0.97      0.99      0.98       785
           7       0.96      0.96      0.96       893
           8       0.99      0.93      0.96       835
           9       0.94      0.95      0.95       838

    accuracy                           0.96      8400
   macro avg       0.97      0.96      0.96      8400
weighted avg       0.97      0.96      0.96      8400

==================================================
Epoch 1/10
WARNING:tensorflow:AutoGraph could not transform <function Model.make_train_function.<locals>.train_function at 0x7c8019c82440> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: closure mismatch, requested ('self', 'step_function'), but source function had ()
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
WARNING: AutoGraph could not transform <function Model.make_train_function.<locals>.train_function at 0x7c8019c82440> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: closure mismatch, requested ('self', 'step_function'), but source function had ()
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
1050/1050 [==============================] - 28s 26ms/step - loss: 0.1945 - accuracy: 0.9427
Epoch 2/10
1050/1050 [==============================] - 23s 22ms/step - loss: 0.0614 - accuracy: 0.9807
Epoch 3/10
1050/1050 [==============================] - 24s 23ms/step - loss: 0.0391 - accuracy: 0.9879
Epoch 4/10
1050/1050 [==============================] - 22s 21ms/step - loss: 0.0246 - accuracy: 0.9919
Epoch 5/10
1050/1050 [==============================] - 25s 24ms/step - loss: 0.0183 - accuracy: 0.9940
Epoch 6/10
1050/1050 [==============================] - 22s 21ms/step - loss: 0.0121 - accuracy: 0.9963
Epoch 7/10
1050/1050 [==============================] - 24s 23ms/step - loss: 0.0094 - accuracy: 0.9970
Epoch 8/10
1050/1050 [==============================] - 22s 21ms/step - loss: 0.0089 - accuracy: 0.9970
Epoch 9/10
1050/1050 [==============================] - 24s 22ms/step - loss: 0.0057 - accuracy: 0.9983
Epoch 10/10
1050/1050 [==============================] - 23s 22ms/step - loss: 0.0063 - accuracy: 0.9978
WARNING:tensorflow:AutoGraph could not transform <function Model.make_predict_function.<locals>.predict_function at 0x7c8019c83010> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: closure mismatch, requested ('self', 'step_function'), but source function had ()
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
WARNING: AutoGraph could not transform <function Model.make_predict_function.<locals>.predict_function at 0x7c8019c83010> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: closure mismatch, requested ('self', 'step_function'), but source function had ()
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
263/263 [==============================] - 3s 9ms/step
263/263 [==============================] - 2s 6ms/step
Model: Convolutional Neural Network
Accuracy: 0.9810714285714286
Classification Report:
               precision    recall  f1-score   support

           0       1.00      0.99      0.99       816
           1       0.99      1.00      0.99       909
           2       0.98      0.98      0.98       846
           3       0.99      0.97      0.98       937
           4       0.98      0.99      0.98       839
           5       0.96      0.97      0.97       702
           6       1.00      0.98      0.99       785
           7       0.99      0.98      0.99       893
           8       0.96      0.98      0.97       835
           9       0.98      0.97      0.98       838

    accuracy                           0.98      8400
   macro avg       0.98      0.98      0.98      8400
weighted avg       0.98      0.98      0.98      8400

==================================================
Model: Gradient Boosting
Accuracy: 0.9417857142857143
Classification Report:
               precision    recall  f1-score   support

           0       0.98      0.98      0.98       816
           1       0.97      0.99      0.98       909
           2       0.94      0.93      0.94       846
           3       0.93      0.91      0.92       937
           4       0.93      0.95      0.94       839
           5       0.93      0.92      0.93       702
           6       0.95      0.96      0.96       785
           7       0.95      0.93      0.94       893
           8       0.93      0.93      0.93       835
           9       0.90      0.92      0.91       838

    accuracy                           0.94      8400
   macro avg       0.94      0.94      0.94      8400
weighted avg       0.94      0.94      0.94      8400

==================================================


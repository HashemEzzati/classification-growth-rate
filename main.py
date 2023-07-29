from sklearn.metrics import confusion_matrix
import numpy as np
from preprocessing import PreProcessing
from models import LinearNN, CNNNetwork
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
import yaml

if __name__ == '__main__':
    with open('./config/experiment_config.yml') as config_file:
        experiment_params = yaml.safe_load(config_file)
    phase = experiment_params['phase']
    save_model_path = experiment_params['save_model_path']
    load_model_path = experiment_params['load_model_path']

    if phase == 'train':
        preproc = PreProcessing('./config')
        preproc.make_clusters()
        train_samples, train_labels, test_samples, test_labels = preproc.create_data()
        normalized_train_samples = preproc.make_normalization(train_samples)
        one_hot_train_labels = preproc.make_one_hot(train_labels)
        normalized_test_samples = preproc.make_normalization(test_samples)
        one_hot_test_labels = preproc.make_one_hot(test_labels)
        # Linear Neural Network
        linearnn = LinearNN(config_dirctory='./config',
                            input_size=train_samples.shape[1],
                            output_size=len(np.unique(train_labels)))
        model = linearnn.create_model()
        model = linearnn.trainer(model, normalized_train_samples, one_hot_train_labels)
        model.save(save_model_path)
        '''
            If you want to create a CNN model pleas comment previous lines after "Linear Neural Network" and 
            uncomment next lines until "Make predictions on the test data"
        '''
        # # Convolutional Neural Network
        # cnn = CNNNetwork(config_dirctory='./config',
        #                  input_size=samples.shape[1],
        #                  output_size=len(np.unique(labels)))
        # model = cnn.create_model()
        # model = cnn.trainer(model, normalized_train_samples, one_hot_train_labels)

        # Make predictions on the test data
        test_pred = model.predict(normalized_test_samples)

        # Create the confusion matrix and classification report
        print(classification_report(test_labels,
                                    test_pred.argmax(axis=1) - (test_pred.shape[1] - 1) / 2,
                                    labels=np.unique(test_labels)))
        print(confusion_matrix(test_labels,
                               test_pred.argmax(axis=1) - (test_pred.shape[1] - 1) / 2))
    if phase == 'test':
        print('It takes a few minutes to do preprocessing...\n')
        preproc = PreProcessing('./config')
        preproc.make_clusters()
        train_samples, train_labels, test_samples, test_labels = preproc.create_data()
        normalized_test_samples = preproc.make_normalization(test_samples)
        one_hot_test_labels = preproc.make_one_hot(test_labels)
        model = load_model(load_model_path)
        test_pred = model.predict(normalized_test_samples)

        # Create the confusion matrix and classification report
        print(classification_report(test_labels,
                                    test_pred.argmax(axis=1) - (test_pred.shape[1] - 1) / 2,
                                    labels=np.unique(test_labels)))
        print(confusion_matrix(test_labels,
                               test_pred.argmax(axis=1) - (test_pred.shape[1] - 1) / 2))

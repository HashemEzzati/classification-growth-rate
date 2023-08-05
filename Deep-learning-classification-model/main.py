from sklearn.metrics import confusion_matrix
import numpy as np
from preprocessing import PreProcessing
from models import LinearNN, CNNNetwork
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
import yaml
from fetch_data import fetch_data
from display import plotter

if __name__ == '__main__':
    with open('./config/experiment_config.yml') as config_file:
        experiment_params = yaml.safe_load(config_file)
    phase = experiment_params['phase']
    save_model_path = experiment_params['save_model_path']
    load_model_path = experiment_params['load_model_path']
    train_start_time = experiment_params['train_start_time']
    train_end_time = experiment_params['train_end_time']
    test_start_time = experiment_params['test_start_time']
    test_end_time = experiment_params['test_end_time']
    train_file_name = experiment_params['train_file_name']
    test_file_name = experiment_params['test_file_name']

    if phase == 'train':
        fetch_data(train_start_time, train_end_time, train_file_name)
        preproc = PreProcessing(data_path='./data/'+train_file_name, phase=phase, config_dirctory='./config')
        preproc.make_clusters()
        train_samples, train_labels = preproc.create_data()
        normalized_train_samples = preproc.make_normalization(train_samples)
        one_hot_train_labels = preproc.make_one_hot(train_labels)
        # # Linear Neural Network
        # linearnn = LinearNN(config_dirctory='./config',
        #                     input_size=train_samples.shape[1],
        #                     output_size=len(np.unique(train_labels)))
        # model = linearnn.create_model()
        # model = linearnn.trainer(model, normalized_train_samples, one_hot_train_labels)
        # model.save(save_model_path)
        '''
            If you want to create a CNN model pleas comment previous lines after "Linear Neural Network" and 
            uncomment next lines until "Make predictions on the test data"
        '''
        # Convolutional Neural Network
        cnn = CNNNetwork(config_dirctory='./config',
                         input_size=train_samples.shape[1],
                         output_size=len(np.unique(train_labels)))
        model = cnn.create_model()
        model = cnn.trainer(model, normalized_train_samples, one_hot_train_labels)
        model.save(save_model_path)

        # Make predictions on the test data
        train_pred = model.predict(normalized_train_samples)

        # Create the confusion matrix and classification report
        print(classification_report(train_labels,
                                    train_pred.argmax(axis=1) - (train_pred.shape[1] - 1) / 2,
                                    labels=np.unique(train_labels)))
        print(confusion_matrix(train_labels,
                               train_pred.argmax(axis=1) - (train_pred.shape[1] - 1) / 2))
    if phase == 'test':
        fetch_data(test_start_time, test_end_time, test_file_name)
        fetch_data(train_start_time, train_end_time, train_file_name)
        train_preproc = PreProcessing(data_path='./data/'+train_file_name, phase='train', config_dirctory='./config')
        model, label_map = train_preproc.make_clusters()
        preproc = PreProcessing(data_path='./data/'+test_file_name, phase=phase, config_dirctory='./config')
        preproc.make_clusters(model, label_map, phase)
        test_samples, test_labels = preproc.create_data()
        normalized_test_samples = preproc.make_normalization(test_samples)
        one_hot_test_labels = preproc.make_one_hot(test_labels)
        model = load_model(load_model_path)
        test_pred = model.predict(normalized_test_samples)

        # Create the confusion matrix and classification report
        print(classification_report(test_labels,
                                    test_pred.argmax(axis=1) - (test_pred.shape[1] - 1) / 2,
                                    labels=np.unique(test_labels)))
        con_mat = confusion_matrix(test_labels,
                                   test_pred.argmax(axis=1) - (test_pred.shape[1] - 1) / 2)
        # print(confusion_matrix(test_labels,
        #                        test_pred.argmax(axis=1) - (test_pred.shape[1] - 1) / 2))
        print(con_mat)

        plotter(con_mat, np.unique(test_labels))

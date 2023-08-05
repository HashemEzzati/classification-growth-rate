<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
</head>
<body>
  <h1>Classification Growth Rate</h1>
  
  <p>In this project, the main focus is on the classification of growth rates. The data is initially clustered based on the growth rates in the hourly time frame, and the number of classes is determined by the user. Each row of data, representing a candle in the hourly time frame, is labeled accordingly.

To create labeled samples, a sliding window of a user-defined size is chosen, and it is considered as a sample. The label for each sample is determined based on the growth rate of the next day or the next candle.

However, since the data is imbalanced, meaning some classes have significantly fewer samples than others, the SMOTE (Synthetic Minority Over-sampling Technique) technique has been employed to address this issue. SMOTE is used to increase the number of samples in the minority class by generating synthetic samples through interpolation between existing samples.

By using SMOTE to augment the data, the imbalance problem is mitigated, and the dataset becomes more balanced. This balanced dataset is then used for training the classification model.
    Considering the highly imbalanced data, during the testing process, the model's performance is evaluated using metrics such as the confusion matrix, accuracy, recall, and F1 score. It should be noted that this model is merely an experimental one, and its results may not necessarily be ideal.</p>
  
  <h2>Table of Contents</h2>
  <ul>
    <li><a href="#Requirements">Requirements</a></li>
    <li><a href="#Configuration">Configuration</a></li>
  </ul>
  
  <h2 id="Requirements">Requirements</h2>
  
  <p>The required version of Bokeh is 3.0.3.</p>
  <p>The required version of tensorflow is 2.11.0.</p>
  <p>The required version of yaml is 0.2.5.</p>
  <p>The required version of pandas is 1.5.2.</p>  
  <p>The required version of numpy is 1.23.1.</p>  
  <p>The required version of sickit-learn is 1.1.1.</p>  
  <p>The required version of imbalanced-learn is 0.11.0.</p>  
  <p>The required version of tqdm is 4.64.1.</p>
  <p>The required version of pymongo is 4.4.1.</p>
  
  <h2 id="Configuration">Configuration</h2>
 
  <p>
In this project, the assumption is that the training model predicts the class of the next day's growth rate in an
    hourly time frame using a sliding window of a user-defined size (default is 100). If necessary,
    you can customize the parameters available in the config folder path.</p>
<p>To configure the parameters for the neural network, experiment, loading data from MongoDB,
  and preprocessing, you need to modify the files in the config folder. Here's a summary of the parameters and their purposes:</p>


<h3>Neural Network Parameters:</h3>
  <ul>
    <li>batch_size: The number of instances in each batch during training.</li>
    <li>epochs: The number of epochs, which represents how many times the model will go through the entire training dataset.</li>
    <li>hidden_size: This parameter is used for linear neural networks and specifies the number of nodes in hidden layer.</li>
  </ul>
  <h3>Experiment Parameters_Training and Testing Time Periods:</h3>
    <ul>
      <li>phase: This parameter should be set as "train" or "test" depending on whether you are training the model or testing it.</li>
      <li>save_model_path: The path where the trained model will be saved.</li>
      <li>load_model_path: The path from where a pre-trained model will be loaded for testing.</li>
      <li>train_start_time: The start time for the training data (e.g., '2023-01-01T00:00:00.000000').</li>
      <li>train_end_time: The end time for the training data (e.g., '2023-03-01T00:00:00.000000').</li>
      <li>train_file_name: The file name for the training samples.</li>
      <li>test_start_time: The start time for the testing data (e.g., '2023-03-01T00:00:00.000000').</li>
      <li>test_end_time: The end time for the testing data (e.g., '2023-06-01T00:00:00.000000').</li>
      <li>test_file_name: The file name for the test samples.</li>
    </ul>
  <h3>Parameters for Loading Data from MongoDB:</h3>
    <ul>
      <li>local: The MongoDB server address, usually set to 'mongodb://127.0.0.1:27017/' for a local server.</li>
      <li>data_base: The name of the database in MongoDB where the data is stored.</li>
      <li>documents: The name of the collection (document) in the database from which data will be retrieved.</li>
    </ul>
  <h3>Preprocessing Config:</h3>
    <ul>
      <li>window_size: The size of the history that the model can see and make decisions based on it.</li>
      <li>cluster_num: The number of classes or clusters for data clustering.</li>
      <li></li>
    </ul>
  <p>Make sure to modify the appropriate configuration files in the config folder with the desired values for each parameter before running your neural network experiment.</p>
  </body>
  </html>

<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
</head>
<body>
  <h1>Classification Growth Rate</h1>
  
  <p>This project focuses on the classification of growth rates. Initially,  Data clustering is performed based on the growth rates in the hourly time frame and the number of classes provided by the user. As the data is imbalanced, SMOTE technique have been employed to increase the existing samples. The training and testing data are preprocessed, and the model chosen by the user (either linear or convolutional) undergoes training. Considering the highly imbalanced data, during the testing process, the model's performance is evaluated using metrics such as the confusion matrix, accuracy, recall, and F1 score. It should be noted that this model is merely an experimental one, and its results may not necessarily be ideal.</p>
  
  <h2>Table of Contents</h2>
  <ul>
    <li><a href="#Requirements">Requirements</a></li>
    <li><a href="#How to use">How to use</a></li>
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
  
  <h2 id="How to use">How to use</h2>
 
  <p>
In this project, the assumption is that the training model predicts the class of the next day's growth rate in an hourly time frame using a sliding window of a user-defined size (default is 300). If necessary, you can customize the parameters available in the config folder path.</p>
  </body>
  </html>

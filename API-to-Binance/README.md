<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
</head>
<body>
  <h1>API to Binance</h1>
  
  <p>In this part of the project, we use Node.js to get API to Binance and store OHLCV data in MongoDB. First, open the terminal in the desired directory, or use the cd command to navigate to the specified path, and then run the command npm init. The required libraries are mentioned in the dependencies section. To install them, use the command npm install library_name in the terminal. To execute the code, use the command node index.js. The OHLCV data will be stored in MongoDB.</p>
  
  <h2>Table of Contents</h2>
  <ul>
    <li><a href="#Requirements">Requirements</a></li>
    <li><a href="#Configuration">Configuration</a></li
  </ul>
  
  <h2 id="Requirements">Requirements</h2>

  <p>The required version of axios is 1.4.0.</p>
  <p>The required version of fs is 0.0.1.</p>
  <p>The required version of js-yaml is 4.1.0.</p>
  <p>The required version of mongoose is 7.4.2.</p>

  <h2 id="Configuration">Configuration</h2>
  <p>To configure the following parameters, modify the config/config.yaml file:</p>
  <ul>
    <li>symbol: Market name. The default is 'BTCUSDT'.</li>
    <li>interval: Time frame. The default is '1h'.</li>
    <li>startTime: Start date. The default is '2020-01-01T00:00:00Z'.</li>
    <li>endTime: End date. The default is '2023-06-01T00:00:00Z'.</li>
    <li>limit: Limitation of the number of OHLCV data per time. The default is 500, and the maximum should be 1500.</li>
    <li>local: Local path to store data. The default is 'mongodb://127.0.0.1:27017'.</li>
    <li>data_base: Database name. The default is '/timeseries_test'.</li>
</ul>

  </body>
  </html>

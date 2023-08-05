// Exporting libraries
const axios = require('axios');
const mongoose = require('mongoose');
const yaml = require('js-yaml');
const fs = require('fs');

// Take data by api to binance
  function api_to_binance(config) {
      return axios.request(config)
          .then((response) => {
              let ohlcvData = response.data.map(item => ({
                  datetime: new Date(item[0]), //Date
                  open: parseFloat(item[1]), // Open
                  high: parseFloat(item[2]), // High
                  low: parseFloat(item[3]), // Low
                  close: parseFloat(item[4]), // Close
                  volume: parseFloat(item[5]), // Volume
              }));
              return ohlcvData;
          })
          .catch((error) => {
              console.log(error);
          });
  }

// General setting
try {
  const currentDirectory = process.cwd();
  const configPath = currentDirectory + '/config/config.yml'
  const config = yaml.load(fs.readFileSync(configPath));
  const symbol = config.market_args.symbol //config.symbol;
  const interval = config.market_args.interval;
  const startTime = new Date(config.market_args.startTime).getTime();
  const endTime = new Date(config.market_args.endTime).getTime();
  const limit = config.market_args.limit;
  const stepInMilliseconds = 60 * 60 * 1000;
  let changeAbleStartTime = new Date(startTime).getTime();
  let newTimeInMilliseconds = startTime + limit * stepInMilliseconds;
  let changeAbleEndTime = newTimeInMilliseconds;
  // MongoDB setting
  const schema = mongoose.Schema({
              datetime: Date,
              open: Number,
              high: Number,
              low: Number,
              close: Number,
              volume: Number,
          })
  const BTCUSDT_1h = mongoose.model('BTCUSDT_1h', schema);
  mongoose.connect(config.mongo_args.local + config.mongo_args.data_base);

  let done = 1
  while (done) {

      const apiUrl = `https://fapi.binance.com/fapi/v1/klines?symbol=${symbol}&interval=${interval}&startTime=${changeAbleStartTime}&endTime=${changeAbleEndTime}&limit=${limit}`;
      let config = {
          method: 'get',
          maxBodyLength: Infinity,
          url: apiUrl,
          headers: {}
      };

      api_to_binance(config).then(ohlcvData => {
          BTCUSDT_1h.insertMany(ohlcvData).then(savedData => {
              console.log('Saved Data', savedData)
          })
              .catch(error => {
                  console.log('Error Occured', error)
              });
      });

      changeAbleStartTime = changeAbleEndTime;
      newTimeInMilliseconds = changeAbleStartTime + limit * stepInMilliseconds;
      changeAbleEndTime = new Date(newTimeInMilliseconds).getTime();
      if (changeAbleEndTime > endTime) {
          if (changeAbleStartTime >= endTime){
              done = 0;
          }
          changeAbleEndTime = endTime;
      }
  }
  console.log('Done');
} catch (error) {
  console.error('Error loading configuration:', error);
}

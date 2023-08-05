from datetime import datetime
import pandas as pd
import pymongo
import yaml
import os


def fetch_data(start_time, end_time, filename):
    with open(os.path.join('./config' + '/fetch_data_config.yml'), 'rb') as config_file:
        params = yaml.safe_load(config_file)
    local = params['local']
    data_base = params['data_base']
    documents = params['documents']

    myclient = pymongo.MongoClient(local)
    mydb = myclient[data_base]
    mycol = mydb[documents]

    date_format = "%Y-%m-%dT%H:%M:%S.%f"
    start_time = datetime.strptime(start_time, date_format)
    end_time = datetime.strptime(end_time, date_format)

    pipline = [{"$match": {"datetime": {"$gte": start_time, "$lte": end_time}}},
               {"$sort": {"datetime": pymongo.ASCENDING}},
               {"$project": {'_id': 0,
                             'datetime': 1,
                             'open': 1,
                             'high': 1,
                             'low': 1,
                             'close': 1,
                             'volume': 1}}]
    data = pd.DataFrame(mycol.aggregate(pipline))
    data.set_index('datetime', inplace=True)
    data.to_csv('./data/' + filename + '.csv', index=True)

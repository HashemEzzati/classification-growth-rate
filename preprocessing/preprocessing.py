import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from collections import deque
import yaml
import os
import tqdm


class PreProcessing:
    def __init__(self, config_dirctory: str = './config'):
        with open(os.path.join(config_dirctory + "/preprocessing_config.yml"), 'rb') as config_file:
            self.params = yaml.safe_load(config_file)
        self.window_size = self.params['window_size']
        self.cluster_num = self.params['cluster_num']
        self.split_date = self.params['split_date']
        self.data = pd.read_csv(self.params['data_path'], index_col='datetime')

    def make_clusters(self):
        for index, row in self.data.iterrows():
            self.data.at[index, 'growth-rate'] = float(round(((row[3] - row[0]) / row[0]) * 100, 2))

        kmeans = KMeans(n_clusters=self.cluster_num,
                        random_state=0).fit(self.data['growth-rate'].values.reshape(-1, 1))
        self.data['labels'] = kmeans.labels_
        labels_mean = np.array([])
        for label in np.unique(kmeans.labels_):
            if len(labels_mean) == 0:
                labels_mean = np.append(labels_mean,
                                        [label, self.data['growth-rate'][self.data['labels'] == label].mean()])
            else:
                labels_mean = np.vstack((labels_mean,
                                        [label, self.data['growth-rate'][self.data['labels'] == label].mean()]))
        indices = np.argsort(labels_mean[:, 1])
        sorted_labels_mean = labels_mean[indices]

        data = self.data.copy()
        for label, new_label in zip(sorted_labels_mean[:, 0],
                                    np.arange(-((self.cluster_num - 1)/2), ((self.cluster_num - 1)/2) + 1, 1)):
            self.data['labels'][data['labels'] == label] = new_label
        self.data.drop(columns=['growth-rate'], inplace=True)

    def create_data(self):
        queue = deque(maxlen=self.window_size)  # * (len(self.data.columns) - 1))
        data = pd.DataFrame(columns=np.arange(0, self.window_size * (len(self.data.columns) - 1) + 1))
        shifted_data = self.data.drop(columns=['labels']).iloc[:-1]
        # shifted_data['labels'] = self.data['labels'].iloc[1:]
        counter = 1
        for index, row in tqdm.tqdm(shifted_data.iterrows()):  # self.data.drop(columns=['labels']).iterrows():
            queue.append(row.values)
            if (counter >= self.window_size) and (counter < len(self.data)-1):
                new_row_data = pd.DataFrame([np.append(queue, self.data.iloc[counter]['labels'])],
                                            columns=data.columns,
                                            index=[index])
                data = pd.concat([data, new_row_data])
                # data.loc[index] = np.append(queue, self.data.iloc[counter]['labels'])
            counter += 1
        data[data.columns[-1]] = data[data.columns[-1]].values.astype(int)
        test_data = data.loc[self.split_date:]
        test_samples = test_data[test_data.columns[:-1]].values
        test_labels = test_data[test_data.columns[-1]].values

        smote = SMOTE(sampling_strategy='auto', random_state=42)

        train_samples, train_labels = smote.fit_resample(data.drop(columns=[data.columns[-1]]).values,
                                                         data[data.columns[-1]].values)
        return train_samples, train_labels, test_samples, test_labels

    @staticmethod
    def make_normalization(data):
        column_norms = np.linalg.norm(data, axis=0)
        return data / column_norms

    @staticmethod
    def make_one_hot(data):
        input_range = [np.unique(data).min(), np.unique(data).max()]
        output_range = [0, len(np.unique(data)) - 1]

        # Perform interpolation to map a to b
        mapped_values = np.interp(data, input_range, output_range)

        unique_values = np.unique(data)
        labels_matrix = np.zeros((len(data), len(unique_values)))
        labels_matrix[np.arange(len(data)), mapped_values.astype(int)] = 1
        return labels_matrix

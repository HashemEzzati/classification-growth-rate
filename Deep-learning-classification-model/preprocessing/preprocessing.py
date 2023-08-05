import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from collections import deque
import yaml
import os
import tqdm


class PreProcessing:
    def __init__(self, data_path, phase, config_dirctory: str = './config'):
        with open(os.path.join(config_dirctory + "/preprocessing_config.yml"), 'rb') as config_file:
            self.params = yaml.safe_load(config_file)
        self.window_size = self.params['window_size']
        self.cluster_num = self.params['cluster_num']
        self.phase = phase
        self.data = pd.read_csv(data_path + '.csv', index_col='datetime')
        print('It takes a few minutes ... . be patient')

    def make_clusters(self, model=None, label_map=None, mode: str = 'train'):
        for index, row in self.data.iterrows():
            self.data.at[index, 'growth-rate'] = float(round(((row[3] - row[0]) / row[0]) * 100, 2))
        if mode == 'train':
            kmeans = KMeans(n_clusters=self.cluster_num, random_state=0)
            kmeans.fit(self.data['growth-rate'].values.reshape(-1, 1))
            labels = kmeans.labels_
            self.data['labels'] = labels
            labels_mean = np.array([])
            for label in np.unique(labels):
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
                                        np.arange(-((self.cluster_num - 1) / 2), ((self.cluster_num - 1) / 2) + 1, 1)):
                self.data['labels'][data['labels'] == label] = new_label
            self.data.drop(columns=['growth-rate'], inplace=True)
            label_map = [[np.unique(self.data['labels'].values)], [sorted_labels_mean[:, 0]]]
            return kmeans, label_map
        elif mode == 'test':
            kmeans = model
            labels = kmeans.predict(self.data['growth-rate'].values.reshape(-1, 1))
            mapping_dict = dict(zip(np.array(label_map[1]).reshape(-1), np.array(label_map[0]).reshape(-1)))

            mapped_labels = [mapping_dict[label] if label in mapping_dict else label for label in labels]
            self.data.drop(columns=['growth-rate'], inplace=True)
            self.data['labels'] = mapped_labels
        else:
            print('Error')
            exit()

    def create_data(self):
        queue = deque(maxlen=self.window_size)
        data = pd.DataFrame(columns=np.arange(0, self.window_size * (len(self.data.columns) - 1) + 1))
        shifted_data = self.data.drop(columns=['labels']).iloc[:-1]
        counter = 1
        for index, row in tqdm.tqdm(shifted_data.iterrows()):
            queue.append(row.values)
            if (counter >= self.window_size) and (counter < len(self.data)-1):
                new_row_data = pd.DataFrame([np.append(queue, self.data.iloc[counter]['labels'])],
                                            columns=data.columns,
                                            index=[index])
                data = pd.concat([data, new_row_data])
            counter += 1
        data[data.columns[-1]] = data[data.columns[-1]].values.astype(int)
        if self.phase == 'train':
            smote = SMOTE(sampling_strategy='auto', random_state=42)

            samples, labels = smote.fit_resample(data.drop(columns=[data.columns[-1]]).values,
                                                 data[data.columns[-1]].values)
            return samples, labels
        elif self.phase == 'test':
            samples = data[data.columns[:-1]].values
            labels = data[data.columns[-1]].values
            return samples, labels
        else:
            print('Thete is an error')
            samples = []
            labels = []
            return samples, labels

    @staticmethod
    def make_normalization(data):
        column_norms = np.linalg.norm(data, axis=0)
        return data / column_norms

    @staticmethod
    def make_one_hot(data):
        input_range = [np.unique(data).min(), np.unique(data).max()]
        output_range = [0, len(np.unique(data)) - 1]

        mapped_values = np.interp(data, input_range, output_range)

        unique_values = np.unique(data)
        labels_matrix = np.zeros((len(data), len(unique_values)))
        labels_matrix[np.arange(len(data)), mapped_values.astype(int)] = 1
        return labels_matrix

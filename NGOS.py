import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
import random
from utils.dataloader import DataLoader
def add_label(X, y):
    data = np.insert(X, X.shape[1], y, axis=1)  # 合并标签
    return data
class NGOS():
    def __init__(self, alpha=1, positive_label=0, negative_label=1):
        super(NGOS, self).__init__()
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.alpha = alpha  # covariance coefficient
        self.N = 0  # the number of minority samples
        self.M = 0  # the number of majority samples
        self.l = 0  # the dimension of input data
        self.min_index = []  # index of minority samples
        self.maj_index = []  # index of majority samples
    def fixed_radius(self, X, y):
        dis_sum = 0.0
        num = len(y)
        for x in X:
            dis = np.linalg.norm(x - X, ord=2, axis=1)
            dis_sum += np.sum(dis)
        R = dis_sum / (num * (num - 1))
        return R
    def candidate_set(self, X, y, sample, R):
        dis_x_train = np.linalg.norm(sample - X, ord=2, axis=1)
        x_candidate_ids = np.where(dis_x_train < R)
        x_candidate = X[x_candidate_ids]
        y_candidate = y[x_candidate_ids]
        return x_candidate, y_candidate

    def probabilistic_anchor_instance_selection(self, I):
        a = [i for i in range(self.N)]  # 少数类样本 的 序号
        gamma = random.choices(a, weights=I, k=1)[0]  # ROULETTE SELECTION FOR THE MINORITY INSTANCES
        return gamma
    def new_instance_generation(self, I, min_sample):
        k = 1
        G = self.M - self.N  # samples need to generate
        new_instances = []
        neigh = NearestNeighbors(n_neighbors=2).fit(min_sample)
        dist_min, indices_min = neigh.kneighbors(min_sample)
        while k <= G:
            selected_index = self.probabilistic_anchor_instance_selection(I)
            anchor = min_sample[selected_index]
            V = np.random.uniform(-1, 1, size=(self.l,))
            # Randomly select a direction originating from the anchor minority instance
            d_0 = np.linalg.norm(anchor - V)
            mu = 0
            sigma = dist_min[selected_index, 1]  # the distance between anchor and its k-nearest minority neighbors
            d_i = self.alpha * sigma * np.random.randn(
                1) + mu  # d_i is a random number generated based on the Gaussian distribution
            r = d_i / d_0
            synthetic_instance = anchor + r * (V - anchor)
            new_instances.append(synthetic_instance)
            k += 1
        return np.array(new_instances)
    def minority_instance_weighting(self, X, y, R):
        C = np.zeros(self.N)  # the density factor
        D = np.zeros(self.N)  # the distance factor
        for i, index in enumerate(self.min_index):
            minority_sample = X[index]
            x_candidate, y_candidate = self.candidate_set(X, y,minority_sample, R)
            maj_num = len(np.where(y_candidate == self.negative_label)[0])
            min_num = len(np.where(y_candidate == self.positive_label)[0])
            pi = maj_num / (maj_num + min_num)
            if pi == 0 or pi == 1:
                C[i] = 0
            else:
                C[i] = -pi * np.log2(pi) - (1 - pi) * np.log2(1 - pi)
            maj_ids = np.where(y_candidate == self.negative_label)

            dist = np.linalg.norm(minority_sample - x_candidate, axis=1)
            dist_all = np.sum(dist)
            dist_maj = np.sum(dist[maj_ids])
            if maj_num == 0 or min_num == 0:
                D[i] = 0
            else:
                D[i] = (dist_maj / maj_num) / ((dist_maj / maj_num) + ((dist_all - dist_maj) / min_num))
        I = C + D  # the information weight

        return self.normalize(I)
    def normalize(self, a):
        a = a.reshape(-1, 1)
        min_max_scaler = preprocessing.MinMaxScaler()  # normalize to 0-1
        a = min_max_scaler.fit_transform(a)
        return a.reshape(1, -1)[0]
    def fit_sample(self, X, y):

        for i in range(len(y)):
            if y[i] == self.positive_label:
                self.min_index.append(i)
            else:
                 self.maj_index.append(i)
        min_sample = X[self.min_index]
        maj_sample = X[self.maj_index]
        self.N = len(min_sample)
        self.M = len(maj_sample)
        self.l = X.shape[1]
        R = self.fixed_radius(X, y)
        I = self.minority_instance_weighting(X, y, R)
        new_instances = self.new_instance_generation(I, min_sample)
        if new_instances.shape[0] == 0:
            Resampled_Data = add_label(X, y)
        else:
            Resampled_Data = np.concatenate((add_label(X, y), add_label(new_instances, self.positive_label)), axis=0)


        #return Resampled_Data
        return Resampled_Data[:, :-1], Resampled_Data[:, -1]
if __name__ == '__main__':
    X, y = DataLoader.get_data('./gdo_exp_datasets/scheme1/abalone19.csv')
    ngos = NGOS()
    balance_X, balance_y = ngos.fit_sample(X, y)
    print(balance_y.shape)
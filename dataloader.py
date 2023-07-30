import numpy as np
import pandas as pd
from GDO import GDO
from sklearn.preprocessing import MinMaxScaler




class DataLoader():


    @staticmethod
    def get_data(data_path, has_header=False):
        #读入数据
        if has_header:
            data = pd.read_csv(data_path)
        else:
            data = pd.read_csv(data_path, header=None)
        X, y = data.iloc[:, :-1].to_numpy(dtype=np.float32), data.iloc[:, -1].to_numpy(dtype=np.float32)


        #归一化
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        # 交叉划分
        return X, y



if __name__ == '__main__':
    X, y = DataLoader.get_data('../keel/wisconsin.csv')
    print(X)
    gdo = GDO()
    x, y = gdo.fit_sample(X, y)


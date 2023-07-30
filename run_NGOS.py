import pandas as pd
import numpy as np
import argparse
from utils.dataloader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import NGOS
from utils.eval import Eval
import os
from tqdm import tqdm

def get_excel_writer(args, file_name):
    os.makedirs(args.result_path, exist_ok=True)
    writer = pd.ExcelWriter(args.result_path + '/'+file_name+'.xlsx', mode='w+', engine='openpyxl')
    return writer
def set_excel_writer(args, writer, alpha, start_colnum):
    for sheet_name in args.sheet_names:
        pd.DataFrame([['alpha='+str(alpha)]]).to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=start_colnum, index=False,header=False)
def train(args, file_name):
    list_dir = os.listdir(args.data_path)
    row = 1

    file_nums = len(list_dir)
    alphas = np.arange(2, 21, 2)/10
    pbar = tqdm(list_dir, total=file_nums)
    pbar.set_description(("%12s  %37s  ") % ('loading...', 'loading...'))
    writer = get_excel_writer(args, file_name)
    #set_excel_writer(args, writer)
    best_writer = pd.ExcelWriter(args.result_path + '/the_best_keel_result_for_NGOS.xlsx', mode='w+', engine='openpyxl')
    for sheet_name in args.sheet_names:
        pd.DataFrame([['', 'alpha', args.algorithm_name]]).to_excel(best_writer, sheet_name=sheet_name, startrow=0, index=False, header=False)
    for dir_name in list_dir:
        colnum = 1
        best_svm_total_f_measure = 0
        best_svm_total_auc = 0
        best_svm_total_g_mean = 0
        best_dt_total_f_measure = 0
        best_dt_total_auc = 0
        best_dt_total_g_mean = 0
        best_rf_total_f_measure = 0
        best_rf_total_auc = 0
        best_rf_total_g_mean = 0
        best_knn_total_f_measure = 0
        best_knn_total_auc = 0
        best_knn_total_g_mean = 0
        best_svm_alpha = 0
        best_dt_alpha = 0
        best_rf_alpha = 0
        best_knn_alpha = 0
        X, y = DataLoader.get_data(os.path.join(args.data_path, dir_name), has_header=False)
        split_num = args.split_num
        cycle_num = args.cycle_num
        dir_name = dir_name.replace('.csv', '')
        for sheet_name in args.sheet_names:
            pd.DataFrame([[dir_name]]).to_excel(writer, sheet_name=sheet_name, startrow=row, startcol=0, index=False, header=False)
        for alpha in alphas:
            set_excel_writer(args, writer, alpha, colnum)
            svm_total_f_measure = 0
            svm_total_auc = 0
            svm_total_g_mean = 0
            dt_total_f_measure = 0
            dt_total_auc = 0
            dt_total_g_mean = 0
            rf_total_f_measure = 0
            rf_total_auc = 0
            rf_total_g_mean = 0
            knn_total_f_measure = 0
            knn_total_auc = 0
            knn_total_g_mean = 0
            for i in range(cycle_num):
                skf = StratifiedKFold(n_splits=split_num, shuffle=True)
                for train_ids, test_ids in skf.split(X, y):
                    X_train, y_train = X[train_ids], y[train_ids]
                    X_test, y_test = X[test_ids], y[test_ids]
                    algorithm = NGOS(alpha=alpha)
                    X_train_balanced, y_train_balanced = algorithm.fit_sample(X_train, y_train)
                    # ===========================svm============================================
                    svm = SVC()
                    svm.fit(X_train_balanced, y_train_balanced)
                    svm_pred = svm.predict(X_test)
                    svm_f_measure, svm_auc, svm_g_mean = Eval.get_metrics(y_test, svm_pred)
                    svm_total_f_measure += svm_f_measure
                    svm_total_auc += svm_auc
                    svm_total_g_mean += svm_g_mean

                    # ==========================dt==============================================
                    dt = DecisionTreeClassifier()
                    dt.fit(X_train_balanced, y_train_balanced)
                    dt_pred = dt.predict(X_test)
                    dt_f_measure, dt_auc, dt_g_mean = Eval.get_metrics(y_test, dt_pred)
                    dt_total_f_measure += dt_f_measure
                    dt_total_auc += dt_auc
                    dt_total_g_mean += dt_g_mean

                    # ==========================rf==============================================
                    rf = RandomForestClassifier()
                    rf.fit(X_train_balanced, y_train_balanced)
                    rf_pred = rf.predict(X_test)
                    rf_f_measure, rf_auc, rf_g_mean = Eval.get_metrics(y_test, rf_pred)
                    rf_total_f_measure += rf_f_measure
                    rf_total_auc += rf_auc
                    rf_total_g_mean += rf_g_mean

                    # ==========================knn=============================================
                    knn = KNeighborsClassifier()
                    knn.fit(X_train_balanced, y_train_balanced)
                    knn_pred = knn.predict(X_test)
                    knn_f_measure, knn_auc, knn_g_mean = Eval.get_metrics(y_test, knn_pred)
                    knn_total_f_measure += knn_f_measure
                    knn_total_auc += knn_auc
                    knn_total_g_mean += knn_g_mean
            svm_total_f_measure /= (split_num * cycle_num)
            svm_total_auc /= (split_num * cycle_num)
            svm_total_g_mean /= (split_num * cycle_num)
            dt_total_f_measure /= (split_num * cycle_num)
            dt_total_auc /= (split_num * cycle_num)
            dt_total_g_mean /= (split_num * cycle_num)
            rf_total_f_measure /= (split_num * cycle_num)
            rf_total_auc /= (split_num * cycle_num)
            rf_total_g_mean /= (split_num * cycle_num)
            knn_total_f_measure /= (split_num * cycle_num)
            knn_total_auc /= (split_num * cycle_num)
            knn_total_g_mean /= (split_num * cycle_num)
            if (svm_total_f_measure + svm_total_auc + svm_total_g_mean) > (best_svm_total_f_measure + best_svm_total_auc + best_svm_total_g_mean):
                best_svm_total_f_measure = svm_total_f_measure
                best_svm_total_auc = svm_total_auc
                best_svm_total_g_mean = svm_total_g_mean
                best_svm_alpha = alpha
            if (dt_total_f_measure + dt_total_auc + dt_total_g_mean) > (best_dt_total_f_measure + best_dt_total_auc + best_dt_total_g_mean):
                best_dt_total_f_measure = dt_total_f_measure
                best_dt_total_auc = dt_total_auc
                best_dt_total_g_mean = dt_total_g_mean
                best_dt_alpha = alpha
            if (rf_total_f_measure + rf_total_auc + rf_total_g_mean) > (best_rf_total_f_measure + best_rf_total_auc + best_rf_total_g_mean):
                best_rf_total_f_measure = rf_total_f_measure
                best_rf_total_auc = rf_total_auc
                best_rf_total_g_mean = rf_total_g_mean
                best_rf_alpha = alpha
            if (knn_total_f_measure + knn_total_auc + knn_total_g_mean) > (best_knn_total_f_measure + best_knn_total_auc + best_knn_total_g_mean):
                best_knn_total_f_measure = knn_total_f_measure
                best_knn_total_auc = knn_total_auc
                best_knn_total_g_mean = knn_total_g_mean
                best_knn_alpha = alpha
            pd.DataFrame([[svm_total_f_measure]]).to_excel(writer, sheet_name='f_measure_svm', startrow=row, startcol=colnum, index=False, header=False)
            pd.DataFrame([[svm_total_auc]]).to_excel(writer, sheet_name='auc_svm', startrow=row, startcol=colnum, index=False, header=False)
            pd.DataFrame([[svm_total_g_mean]]).to_excel(writer, sheet_name='g_mean_svm', startrow=row, startcol=colnum,  index=False, header=False)
            pd.DataFrame([[dt_total_f_measure]]).to_excel(writer, sheet_name='f_measure_dt', startrow=row, startcol=colnum, index=False, header=False)
            pd.DataFrame([[dt_total_auc]]).to_excel(writer, sheet_name='auc_dt', startrow=row, startcol=colnum, index=False, header=False)
            pd.DataFrame([[dt_total_g_mean]]).to_excel(writer, sheet_name='g_mean_dt', startrow=row, startcol=colnum, index=False, header=False)
            pd.DataFrame([[rf_total_f_measure]]).to_excel(writer, sheet_name='f_measure_rf', startrow=row, startcol=colnum,  index=False, header=False)
            pd.DataFrame([[rf_total_auc]]).to_excel(writer, sheet_name='auc_rf', startrow=row, index=False, startcol=colnum, header=False)
            pd.DataFrame([[rf_total_g_mean]]).to_excel(writer, sheet_name='g_mean_rf', startrow=row, index=False, startcol=colnum,  header=False)
            pd.DataFrame([[knn_total_f_measure]]).to_excel(writer, sheet_name='f_measure_knn', startrow=row, index=False, startcol=colnum, header=False)
            pd.DataFrame([[knn_total_auc]]).to_excel(writer, sheet_name='auc_knn', startrow=row, index=False, startcol=colnum, header=False)
            pd.DataFrame([[knn_total_g_mean]]).to_excel(writer, sheet_name='g_mean_knn', startrow=row, index=False, startcol=colnum, header=False)
            writer.save()
            colnum += 1
        pd.DataFrame([[dir_name, best_svm_alpha, best_svm_total_f_measure]]).to_excel(best_writer, sheet_name='f_measure_svm', startrow=row, index=False, header=False)
        pd.DataFrame([[dir_name, best_svm_alpha, best_svm_total_auc]]).to_excel(best_writer, sheet_name='auc_svm', startrow=row, index=False, header=False)
        pd.DataFrame([[dir_name, best_svm_alpha, best_svm_total_g_mean]]).to_excel(best_writer, sheet_name='g_mean_svm', startrow=row, index=False, header=False)
        pd.DataFrame([[dir_name, best_dt_alpha, best_dt_total_f_measure]]).to_excel(best_writer, sheet_name='f_measure_dt', startrow=row, index=False, header=False)
        pd.DataFrame([[dir_name, best_dt_alpha, best_dt_total_auc]]).to_excel(best_writer, sheet_name='auc_dt', startrow=row, index=False, header=False)
        pd.DataFrame([[dir_name, best_dt_alpha, best_dt_total_g_mean]]).to_excel(best_writer, sheet_name='g_mean_dt', startrow=row, index=False, header=False)
        pd.DataFrame([[dir_name, best_rf_alpha, best_rf_total_f_measure]]).to_excel(best_writer, sheet_name='f_measure_rf', startrow=row, index=False, header=False)
        pd.DataFrame([[dir_name, best_rf_alpha, best_rf_total_auc]]).to_excel(best_writer, sheet_name='auc_rf', startrow=row, index=False, header=False)
        pd.DataFrame([[dir_name, best_rf_alpha, best_rf_total_g_mean]]).to_excel(best_writer, sheet_name='g_mean_rf', startrow=row, index=False, header=False)
        pd.DataFrame([[dir_name, best_knn_alpha, best_knn_total_f_measure]]).to_excel(best_writer, sheet_name='f_measure_knn', startrow=row, index=False, header=False)
        pd.DataFrame([[dir_name, best_knn_alpha, best_knn_total_auc]]).to_excel(best_writer, sheet_name='auc_knn', startrow=row, index=False, header=False)
        pd.DataFrame([[dir_name, best_knn_alpha, best_knn_total_g_mean]]).to_excel(best_writer, sheet_name='g_mean_knn', startrow=row, index=False, header=False)
        best_writer.save()
        row += 1
        pbar.set_description((f"%8s/{file_nums}  %35s is finished ") % (row-1, dir_name))
        pbar.update(1)
    best_writer.close()
    writer.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, help='result path')
    parser.add_argument('--data_path', type=str, default='./keel', help='data path')
    parser.add_argument('--algorithm_name', type=str, default='NGOS')
    parser.add_argument('--sheet_names', type=list, default=['f_measure_svm', 'auc_svm', 'g_mean_svm', 'f_measure_dt', 'auc_dt', 'g_mean_dt', 'f_measure_rf', 'auc_rf', 'g_mean_rf', 'f_measure_knn', 'auc_knn', 'g_mean_knn'])
    parser.add_argument('--split_num', type=int, default=5)
    parser.add_argument('--cycle_num', type=int, default=5)
    args = parser.parse_args()
    train(args, 'NGOS_find_the_best_for_keel')
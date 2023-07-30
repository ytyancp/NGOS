
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from imblearn.metrics import geometric_mean_score





class Eval:
    @staticmethod
    def get_metrics(trueset, preset):
        # cm = confusion_matrix(trueset, preset)          # 得到混淆矩阵
        # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # tp = cm[0, 0]
        # fn = cm[0, 1]
        # fp = cm[1, 0]
        # tn = cm[1, 1]
        tp, fn, fp, tn = confusion_matrix(trueset, preset).ravel()  # 0是少数类的情况
        # tn, fp, fn, tp = confusion_matrix(trueset, preset).ravel() # 1是少数类的情况
        # tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()    # 1是少数类的情况

        # tp_rate = tp / (tp + fn)
        # fp_rate = fp / (fp + tn)
        # acc_temp = (tp + tn) / (tp + tn + fp + fn)
        # precision_temp = tp / (tp + fp)
        # recall_temp = tp / (tp + fn)
        # f_measure_temp = (2 * tp) / (2 * tp + fp + fn)
        f_measure_temp = f1_score(trueset, preset, pos_label=0)
        # auc_temp = (1 + tp_rate - fp_rate) / 2
        auc_temp = roc_auc_score(trueset, preset)
        # precision = precision_score(trueset, preset, pos_label=0)
        # recall = recall_score(trueset, preset, pos_label=0)
        # f_measure_temp = 2 * precision * recall / (precision + recall)
        # g_mean_temp = np.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))
        g_mean_temp = geometric_mean_score(trueset, preset, pos_label=0)

        return f_measure_temp, auc_temp, g_mean_temp





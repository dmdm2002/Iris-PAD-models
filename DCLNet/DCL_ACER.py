import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Detection Rate')
    plt.ylabel('True Detection Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()

path_list = [['warsaw/1-fold/origianl/warsaw_1-fold_label_9.csv', 'warsaw/1-fold/original/warsaw_1-fold_prov_9.csv'],
             ['warsaw/2-fold/original/warsaw_2-fold_label_5.csv', 'warsaw/2-fold/original/warsaw_2-fold_prov_5.csv']]

full_apcer = 0
full_bpcer = 0
count = 1
for path in path_list:
    print(path[0], path[1])
    score = pd.read_csv(path[1])
    label = pd.read_csv('warsaw/1-fold/origianl/warsaw_1-fold_label_9.csv')

    score = np.array(score.drop('Unnamed: 0', axis=1).astype('float32'))
    label = np.array(label.drop('Unnamed: 0', axis=1).astype('float32'))

    predict_list = []
    for i in range(len(score)):
        if score[i] >= 0.5:
            predict_list.append([i, label[i], 1])
        else:
            predict_list.append([i, label[i], 0])

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(predict_list)):
        if predict_list[i][1] == 1 and predict_list[i][2] == 0:
            fp = fp + 1
        elif predict_list[i][1] == 1 and predict_list[i][2] == 1:
            tp = tp + 1
        elif predict_list[i][1] == 0 and predict_list[i][2] == 1:
            fn = fn + 1
        elif predict_list[i][1] == 0 and predict_list[i][2] == 0:
            tn = tn + 1

    apcer = fn / (tp + fn)
    bpcer = fp / (fp + tn)

    print(tp)
    print(fp)
    print(tn)
    print(fn)

    count = count + 1

    print(f"APCER : {apcer}")
    print(f"BPCER : {bpcer}")

    full_apcer = full_apcer + apcer
    full_bpcer = full_bpcer + bpcer

a = full_apcer / 2
b = full_bpcer / 2
c = (a + b) / 2

print(f"avg APCER : {a * 100}")
print(f"avg BPCER : {b * 100}")
print(f"avg BPCER : {c * 100}")
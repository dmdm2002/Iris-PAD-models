import pandas as pd
import numpy as np
import os

phases = ['train', 'test']
data = []
for phase in phases:
    if phase == 'train':
        root = 'Z:/Iris_dataset/Warsaw_labeling_iris_data/innerclass/Proposed/1-fold/A/iris'
    else:
        root = 'Z:/Iris_dataset/Warsaw_labeling_iris_data/innerclass/Proposed/1-fold/B/iris'

    classes = ['fake', 'live']

    for class_type in classes:
        folder_path = f'{root}/{class_type}'

        images = os.listdir(folder_path)

        for image in images:
            data.append([phase, class_type, image])


data = np.array(data)
data_df = pd.DataFrame(data)
print(data_df)

data_df.to_csv('./Warsaw_1-fold_test_train_split_new.csv', index=False)
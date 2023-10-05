import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from sklearn import svm
import time
import datetime
import pandas as pd

###########################################################################################
#                                        set gpu
###########################################################################################
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate xGB of memory on the first GPU
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

def ExtractionFunc(model):
    return tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('svm_output_layer').output)

###########################################################################################
#                                       set dataset
###########################################################################################

label = ['Fake', 'Live']
path = 'Z:/1st/Iris_dataset/nd_labeling_iris_data'

train_path = f'{path}/CycleGAN/1-fold/A'
test_path = f'{path}/CycleGAN/1-fold/B'

train_data = os.listdir(train_path)
test_data = os.listdir(test_path)

# original data
# batchsz = 1
# traincnt = 5182
# testcnt = 5154
# valcnt = 1191

# cyclegan data ND 2-fold
batchsz = 1
traincnt = 5018
testcnt = 4554
valcnt = 1036

# cyclegan data ND 2-fold
# batchsz = 2
# traincnt = 5018
# testcnt = 4554
# valcnt = 1036

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_path, target_size=(224, 224), batch_size=1)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = train_datagen.flow_from_directory(test_path, target_size=(224, 224), batch_size=1)

input_shape = (224, 224, 3)

# set Model
baseModel = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# layer[52] -> Second DenseBlock pooling layer (pool2_pool)
x = baseModel.get_layer(baseModel.layers[52].name).output
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(512)(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(128, name='svm_output_layer')(x)
x = tf.keras.layers.Dense(2, activation='softmax')(x)

DCLNet = tf.keras.models.Model(inputs=baseModel.input, outputs=x)

for layer in DCLNet.layers[:27]:
    layer.trainable = False


optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9)
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
loss_mean = tf.keras.metrics.Mean()

# set CheckPoint
# iterator = iter(train_generator)
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=DCLNet)
# ckpt_path = 'E:/backup/ckp/nd/Ablation/DCLNet/Gradient_tap/2-fold'
# manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=None)

# set Tensorbard
# train_log_dir = './logs/Grdient_tap/nd/2-fold'
# train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# @tf.function
# def train_step(x, y):
#     with tf.GradientTape() as tape:
#         logits = model(x, training=True)
#         loss_value = loss_fn(y, logits)
#
#     grads = tape.gradient(loss_value, DCLNet.trainable_variables)
#     optimizer.apply_gradients(zip(grads, DCLNet.trainable_variables))
#     # train_acc_metric.update_state(y, logits)
#     loss_mean.update_state(loss_value)
#
#     return loss_value

@tf.function
def train_step(x, y):
    val_logits = model(x, training=False)

    return val_logits


epochs = 1
temp = []
for epoch in range(16, 17):
    iris_ckp_path = f"Z:/1st/backup/ckp/nd/Ablation/DCLNet/Gradient_tap/1-fold/ckpt-{epoch}"
    ckpt.restore(iris_ckp_path)
    model = ExtractionFunc(DCLNet)

    predict_list = []
    for step in range(traincnt):
        img, label = next(train_generator)
        logits = train_step(img, label)
        feature = logits.numpy()

        label = np.array([np.argmax(label)])
        # print(label)
        data_arr = np.append(feature[0], label)
        predict_list.append(data_arr)


    predict_list = np.array(predict_list)

    label = predict_list[:,128:]
    feature = predict_list[:,:128]

    clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
    clf.fit(feature, np.ravel(label,order='C'))

    ###################################################################

    print(f'Epoch {epoch} : start testing')
    predict_list = []
    for step in range(6):
        img, label = next(test_generator)
        start = int(round(time.time() * 1000))
        logits = train_step(img, label)
        feature = logits.numpy()

        label = np.array([np.argmax(label)])
        data_arr = np.append(feature[0], label)

        score = clf.score(feature, np.ravel(label, order='C'))
        end = int(round(time.time() * 1000))

        print("milli second : ", end - start)

    # print(f'Epoch {epoch} : start testing')
    # predict_list = []
    # start = int(round(time.time() * 1000))
    # for step in range(6):
    #     img, label = next(test_generator)
    #     logits = train_step(img, label)
    #     feature = logits.numpy()
    #
    #     label = np.array([np.argmax(label)])
    #     data_arr = np.append(feature[0], label)
    #     predict_list.append(data_arr)
    #
    # predict_list = np.array(predict_list)
    #
    # label = predict_list[:, 128:]
    # feature = predict_list[:, :128]
    #
    # score = clf.score(feature, np.ravel(label,order='C'))
    # end = int(round(time.time() * 1000))
    #
    # print("milli second : ", end - start)
    # temp.append(end-start)

    # print(score)
    #
    # predict = clf.predict(feature)
    # tp = 0
    # fp = 0
    # tn = 0
    # fn = 0
    #
    # for i in range(len(predict)):
    #     if predict[i] == 1 and label[i] == 0:
    #         fp = fp + 1
    #     elif predict[i] == 1 and label[i] == 1:
    #         tn = tn + 1
    #     elif predict[i] == 0 and label[i] == 1:
    #         fn = fn + 1
    #     elif predict[i] == 0 and label[i] == 0:
    #         tp = tp + 1
    #
    # apcer = fn / (tp + fn)
    # bpcer = fp / (fp + tn)
    #
    # print(tp)
    # print(fp)
    # print(tn)
    # print(fn)
    #
    # # print(6 / 2509)
    # print(f"APCER : {apcer}")
    # print(f"BPCER : {bpcer}")
    # # clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
    # # clf.fit(feature, label)


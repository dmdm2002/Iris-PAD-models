import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sklearn
import numpy as np
import os
import datetime
import time

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

###########################################################################################
#                                       set dataset
###########################################################################################

label = ['Fake', 'Live']
path = 'Z:/Iris_dataset/Warsaw_labeling_iris_data/innerclass'

train_path = f'{path}/CycleGAN/2-fold/A'
test_path = f'{path}/CycleGAN/2-fold/B'

train_data = os.listdir(train_path)
test_data = os.listdir(test_path)

# original data
batchsz = 2
traincnt = 5154
testcnt = 5182
valcnt = 1191

# cyclegan data ND 2-fold
# batchsz = 2
# traincnt = 4550
# testcnt = 5018
# valcnt = 1036

# cyclegan data ND 2-fold
# batchsz = 2
# traincnt = 5018
# testcnt = 4554
# valcnt = 1036

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_path, target_size=(224, 224), batch_size=batchsz)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = train_datagen.flow_from_directory(test_path, target_size=(224, 224), batch_size=batchsz)

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
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
test_acc_metric = tf.keras.metrics.CategoricalAccuracy()

# set CheckPoint
# iterator = iter(train_generator)
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=DCLNet)
ckpt_path = 'Z:/backup/ckp/warsaw/Ablation/DCLNet/Gradient_tap/2-fold'
manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=None)

# set Tensorbard
train_log_dir = './logs/Grdient_tap/warsaw_2/2-fold'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = DCLNet(x, training=True)
        loss_value = loss_fn(y, logits)

    grads = tape.gradient(loss_value, DCLNet.trainable_variables)
    optimizer.apply_gradients(zip(grads, DCLNet.trainable_variables))
    train_acc_metric.update_state(y, logits)
    loss_mean.update_state(loss_value)

    return loss_value

@tf.function
def test_step(x, y):
    val_logits = DCLNet(x, training=False)


epochs = 1
for epoch in range(epochs):

    for step in range(testcnt//batchsz):
        img, label = next(test_generator)
        loss_val = train_step(img, label)

        if step % 100 == 0 or (testcnt//batchsz) % step == 0:
            save_path = manager.save()
            print(
                "Traning Loss at step %d: %.4f"
                % (step, float(loss_val))
            )

    result_loss = loss_mean.result()
    train_acc = train_acc_metric.result()
    # save_path = manager.save()

    print(f"LOSS : {result_loss}")
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    # for i in range(testcnt//2):
    #     iris_img, iris_uppper_img, iris_lower_img, iris_label = next(test_generator)
    #     inputs = [iris_img, iris_uppper_img, iris_lower_img]
    #     test_step(inputs, iris_label)
    #
    # test_acc = test_acc_metric.result()
    # test_acc_metric.reset_states()
    # loss_mean.reset_states()
    #
    # print("Validation acc: %.4f" % (float(test_acc),))
    #
    # with train_summary_writer.as_default():
    #     tf.summary.scalar('loss', result_loss, step=epoch)
    #     tf.summary.scalar('Accuracy', train_acc, step=epoch)
    #     tf.summary.scalar('test_Accuracy', test_acc, step=epoch)
    #
    # train_acc_metric.reset_states()


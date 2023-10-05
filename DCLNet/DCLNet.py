import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import datetime

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
path = 'E:/Iris_dataset/nd_labeling_iris_data'

train_path = f'{path}/CycleGAN/2-fold/A'
test_path = f'{path}/CycleGAN/2-fold/B'

train_data = os.listdir(train_path)
test_data = os.listdir(test_path)

# original data
# batchsz = 2
# traincnt = 5154
# testcnt = 5182
# valcnt = 1191

# cyclegan data
batchsz = 2
traincnt = 4554
testcnt = 5018
valcnt = 1036

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_path, target_size=(224, 224), batch_size=batchsz)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = train_datagen.flow_from_directory(test_path, target_size=(224, 224), batch_size=batchsz)

input_shape = (224, 224, 3)

###########################################################################################
#                                       Model
###########################################################################################
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

# optim = tf.keras.optimizers.Adam(learning_rate=0.0001)
optim = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9)
DCLNet.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

###########################################################################################
#                                   set tensorboard
###########################################################################################

log_dir = "./logs/fit/nd/Ablation/DCLNet/1-fold-sixth"
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

###########################################################################################
#                                       set ckp
###########################################################################################

ckp_path = "E:/backup/ckp/nd/Ablation/DCLNet/2-fold-sixth/ckp-{epoch:04d}.ckpt"
ckp_dir = os.path.dirname(ckp_path)

ckp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckp_path, verbose=1, save_weights_only=True)

###########################################################################################
#                                       test call back
###########################################################################################


# class step_test(tf.keras.callbacks.Callback):
#
#     def __init__(self, train_generator):
#         self.generator = train_generator
#
#     def on_epoch_end(self, epoch, logs=None):
#         acc = self.model.evaluate(self.generator, verbose=1)
#         print('===============================================')
#         print(f'{epoch+1} : {acc}')
#         print('===============================================')
#
#
# testing = step_test(train_generator)

###########################################################################################
#                                       training\
###########################################################################################

epochs = 30


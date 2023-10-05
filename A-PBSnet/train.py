import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from attention import spatial_attention
import os

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

# label = ['Fake', 'Live']
# path = 'E:/Iris_dataset/nd_labeling_iris_data'
#
# train_path = f'{path}/CycleGAN/2-fold/A'
# test_path = f'{path}/CycleGAN/2-fold/B'
#
# train_data = os.listdir(train_path)
# test_data = os.listdir(test_path)
#
# # original data
# # batchsz = 2
# # traincnt = 5154
# # testcnt = 5182
# # valcnt = 1191
#
# # cyclegan data
# batchsz = 2
# traincnt = 4554
# testcnt = 5018
# valcnt = 1036
#
# train_datagen = ImageDataGenerator(rescale=1./255)
# train_generator = train_datagen.flow_from_directory(train_path, target_size=(224, 224), batch_size=batchsz)
#
# test_datagen = ImageDataGenerator(rescale=1./255)
# test_generator = train_datagen.flow_from_directory(test_path, target_size=(224, 224), batch_size=batchsz)
#
# input_shape = (224, 224, 3)

###########################################################################################
#                                       Model
###########################################################################################

baseModel = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
output_1 = baseModel.get_layer('pool1 ').output
output_2 = baseModel.get_layer('conv2_block1_concat ').output
output_3 = baseModel.get_layer('conv2_block2_concat ').output

# output_1 = spatial_attention(output_1)
# Model_1 = tf.keras.Model(inputs=baseModel.input, outputs=output_1)
#
# print(Model_1.summary())
# # output_2


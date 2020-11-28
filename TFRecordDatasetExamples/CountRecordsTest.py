import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

import tfds_util
import tensorflow as tf
import numpy as np
import random
import unittest


def createDataset(filename:str, count:int):
  num_outputs = 10
  writer = tf.io.TFRecordWriter(filename)
  
  for i in range(count):
  
    img = np.random.rand(40,40)
    label = tf.keras.utils.to_categorical(random.randint(0,num_outputs-1),num_outputs)
    feature = {
      'raw': tfds_util.float_list_feature(img.flatten()),
      'height': tfds_util.int64_feature(img.shape[0]),
      'width' : tfds_util.int64_feature(img.shape[1]),
      'label': tfds_util.float_list_feature(label),
      'filename' : tfds_util.bytes_feature(bytes('some_file.png','utf-8'))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
          
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
  
  writer.close()

class TestCountRecords(unittest.TestCase):
  
  def test_countSingleDataset(self):
    filename = 'test_file.tfrecords'
    createDataset(filename,300)
    ds = tf.data.TFRecordDataset(filename)
    self.assertEqual(300,tfds_util.countRecords(ds))

if __name__ == "__main__":
    # execute only if run as a script
    print(tf.__version__)
    unittest.main()
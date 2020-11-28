import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

import tfds_util
import tensorflow as tf
import numpy as np
import random
import unittest
import os

def createDataset(filename:str, count:int, close:bool):
  num_outputs = 10
  writer = tfds_util.TFRecordWriterExtended(filename)
  
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
  
  if(close):
    writer.close()

class TestMetaData(unittest.TestCase):
  def setUp(setup):
    files = [
      'test_file.tfrecords',
      'test_file.tfrecords.meta'
      'test_file2.tfrecords'
      'test_file2.tfrecords.meta'
    ]
    for f in files:
      if(os.path.exists(f)):
        os.remove(f)
    

  def test_countSingleDataset(self):
    filename = 'test_file.tfrecords'
    createDataset(filename,300,True)
    ds = tfds_util.TFRecordDatasetExtended(filename)
    self.assertEqual(300,ds.recordCount())

  def test_closeNotCalled(self):
    filename = 'test_file.tfrecords'
    createDataset(filename,300,False)
    ds = tfds_util.TFRecordDatasetExtended(filename)
    self.assertEqual(300,ds.recordCount())

  def test_countMultipleDatasets(self):
    filename1 = 'test_file.tfrecords'
    filename2 = 'test_file2.tfrecords'
    createDataset(filename1,300,False)
    createDataset(filename2,300,False)
    ds = tfds_util.TFRecordDatasetExtended([filename1,filename2])
    self.assertEqual(600,ds.recordCount())

if __name__ == "__main__":
    # execute only if run as a script
    print(tf.__version__)
    unittest.main()
import tensorflow as tf
from tensorflow import keras
import json

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def countRecords(ds:tf.data.Dataset):
  count = 0

  if tf.executing_eagerly():
    # TF v2 or v1 in eager mode
    for r in ds:
      count = count+1
  else:  
    # TF v1 in non-eager mode
    iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
    next_batch = iterator.get_next()
    with tf.compat.v1.Session() as sess:
      try:
        while True:
          sess.run(next_batch)
          count = count+1    
      except tf.errors.OutOfRangeError:
        pass
  
  return count


class TFRecordWriterExtended(tf.io.TFRecordWriter):
  recordCount = 0
  # filename
  # writer

  def __init__(self,path,options=None):
    self.filename = path
    tf.io.TFRecordWriter.__init__(self,path,options)
  
  def __del__(self):
    self.close()

  def write(self,record):
    tf.io.TFRecordWriter.write(self,record)
    self.recordCount += 1

  def close(self):
    tf.io.TFRecordWriter.close(self)
    self.writeMetadata()

  def getMetadata(self):
    # Add more metadata as necessary
    return { 'recordCount':self.recordCount }

  def writeMetadata(self):
    metadataFile = self.filename + '.meta'
    data = self.getMetadata()
    with open(metadataFile,'w') as outfile:
      json.dump(data,outfile)

class TFRecordDatasetExtended(tf.data.TFRecordDataset):

  def __init__(self,filenames, compression_type=None, buffer_size=None, num_parallel_reads=None):
    self.filenames = filenames
    tf.data.TFRecordDataset.__init__(self,filenames, compression_type=None, buffer_size=None, num_parallel_reads=None)

  def recordCount(self):
    count = 0
    if(isinstance(self.filenames,str)):
      return self.recordCountForFile(self.filenames)
    for f in self.filenames:
      count += self.recordCountForFile(f)
    return count

  def recordCountForFile(self,filename):
    metadataFile = filename + '.meta'
    with open(metadataFile,'r') as metafile:
      data = json.load(metafile)
      return data['recordCount']
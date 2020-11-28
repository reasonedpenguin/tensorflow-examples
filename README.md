# Setup environment

```bash
virtualenv --system-site-packages -p python3 ./venv
source ./venv/bin/activate
pip install --upgrade pip
pip install --upgrade tensorflow
```
Or if you want to use TensorFlow version 1
```bash
pip install --upgrade tensorflow==1.15.0
```

# TFRecordDataset examples

So far just a function to count the number of records in a dataset. This works in TF 1 and 2. See the full write up at [https://www.rustyrobotics.com/posts/tensorflow/tfdataset-record-count/](https://www.rustyrobotics.com/posts/tensorflow/tfdataset-record-count/)

```bash 
source ./venv/bin/activate
python TFRecordDatasetExamples/CountRecordsTest.py
```
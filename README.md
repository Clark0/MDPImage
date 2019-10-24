# MDPImage
This project depends on tensorflow object detection API. Need to follow the [installation guide|https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md]

```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
pip install -r requirements.txt
```

## Run server
```
python gRPC/server.py
```
and then can start client
```
python gRPC/client.py
```

## Test video stream
```
python detection_stream.py
```

## Test single image inference
```
python detection.py
```

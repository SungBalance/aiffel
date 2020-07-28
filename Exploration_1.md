```python
!pip install pillow   

from PIL import Image
import os, glob
import numpy as np
```

    Requirement already satisfied: pillow in /home/aiffel/anaconda3/envs/aiffel/lib/python3.7/site-packages (7.2.0)


# 1. 데이터 준비


```python
import os
def resize(image_dir_path):
  print("이미지 디렉토리 경로: ", image_dir_path)
  images=glob.glob(image_dir_path + "/*.jpg")  

  # 파일마다 모두 28x28 사이즈로 바꾸어 저장합니다.
  target_size=(28,28)
  for img in images:
    old_img=Image.open(img)
    new_img=old_img.resize(target_size,Image.ANTIALIAS)
    new_img.save(img,"JPEG")
```


```python
def load_data(img_path, number_of_data):
    # 가위 : 0, 바위 : 1, 보 : 2
    # 가위바위보 이미지 개수 총합에 주의하세요.
    img_size=28
    color=3
    #이미지 데이터와 라벨(가위 : 0, 바위 : 1, 보 : 2) 데이터를 담을 행렬(matrix) 영역을 생성합니다.
    imgs=np.zeros(number_of_data*img_size*img_size*color,dtype=np.int32).reshape(number_of_data,img_size,img_size,color)
    labels=np.zeros(number_of_data,dtype=np.int32)

    idx=0
    for file in glob.iglob(img_path+'/scissor/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=0   # 가위 : 0
        idx=idx+1

    for file in glob.iglob(img_path+'/rock/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=1   # 바위 : 1
        idx=idx+1       
    
    for file in glob.iglob(img_path+'/paper/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=2   # 보 : 2
        idx=idx+1
        
    print("데이터셋의 이미지 개수는",idx,"입니다.")
    return imgs, labels
```


```python
# resize trainset
resize(os.getenv("HOME") + "/aiffel/rock_scissor_paper/scissor")
resize(os.getenv("HOME") + "/aiffel/rock_scissor_paper/rock")
resize(os.getenv("HOME") + "/aiffel/rock_scissor_paper/paper")

# resize testset
resize(os.getenv("HOME") + "/aiffel/rock_scissor_paper/test/scissor")
resize(os.getenv("HOME") + "/aiffel/rock_scissor_paper/test/rock")
resize(os.getenv("HOME") + "/aiffel/rock_scissor_paper/test/paper")
```


```python
# load trainset
image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper"
(x_train, y_train)=load_data(image_dir_path, 574)

x_train = x_train/255.0 # 입력은 0~1 사이의 값으로 정규화
print("x_train shape: {}".format(x_train.shape))
print("y_train shape: {}".format(y_train.shape))

# load testset
image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/test"
(x_test, y_test)=load_data(image_dir_path, 155)

x_test = x_test/255.0 # 입력은 0~1 사이의 값으로 정규화
print("x_test shape: {}".format(x_test.shape))
print("y_test shape: {}".format(y_test.shape))
```

    데이터셋의 이미지 개수는 574 입니다.
    x_train shape: (574, 28, 28, 3)
    y_train shape: (574,)
    데이터셋의 이미지 개수는 155 입니다.
    x_test shape: (155, 28, 28, 3)
    y_test shape: (155,)


# 2. 네트워크 설계 및 학습


```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

n_channel_1=128
n_channel_2=256
n_dense=128
n_train_epoch=20

model=keras.models.Sequential()
model.add(keras.layers.Conv2D(n_channel_1, (4,4), activation='relu', input_shape=(28,28,3)))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(n_channel_2, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(n_dense, activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))

model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 25, 25, 128)       6272      
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 12, 12, 128)       0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 10, 10, 256)       295168    
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 5, 5, 256)         0         
    _________________________________________________________________
    flatten (Flatten)            (None, 6400)              0         
    _________________________________________________________________
    dense (Dense)                (None, 128)               819328    
    _________________________________________________________________
    dense_1 (Dense)              (None, 3)                 387       
    =================================================================
    Total params: 1,121,155
    Trainable params: 1,121,155
    Non-trainable params: 0
    _________________________________________________________________



```python
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=n_train_epoch)
```

    Epoch 1/20
    18/18 [==============================] - 2s 126ms/step - loss: 1.1055 - accuracy: 0.3571
    Epoch 2/20
    18/18 [==============================] - 0s 3ms/step - loss: 1.0313 - accuracy: 0.4965
    Epoch 3/20
    18/18 [==============================] - 0s 3ms/step - loss: 0.8753 - accuracy: 0.5993
    Epoch 4/20
    18/18 [==============================] - 0s 3ms/step - loss: 0.7182 - accuracy: 0.7230
    Epoch 5/20
    18/18 [==============================] - 0s 3ms/step - loss: 0.5994 - accuracy: 0.7979
    Epoch 6/20
    18/18 [==============================] - 0s 3ms/step - loss: 0.4833 - accuracy: 0.8310
    Epoch 7/20
    18/18 [==============================] - 0s 2ms/step - loss: 0.3882 - accuracy: 0.8868
    Epoch 8/20
    18/18 [==============================] - 0s 2ms/step - loss: 0.3044 - accuracy: 0.8955
    Epoch 9/20
    18/18 [==============================] - 0s 2ms/step - loss: 0.2600 - accuracy: 0.9268
    Epoch 10/20
    18/18 [==============================] - 0s 2ms/step - loss: 0.2025 - accuracy: 0.9303
    Epoch 11/20
    18/18 [==============================] - 0s 2ms/step - loss: 0.1740 - accuracy: 0.9443
    Epoch 12/20
    18/18 [==============================] - 0s 2ms/step - loss: 0.1499 - accuracy: 0.9495
    Epoch 13/20
    18/18 [==============================] - 0s 3ms/step - loss: 0.1294 - accuracy: 0.9669
    Epoch 14/20
    18/18 [==============================] - 0s 3ms/step - loss: 0.0963 - accuracy: 0.9721
    Epoch 15/20
    18/18 [==============================] - 0s 3ms/step - loss: 0.0763 - accuracy: 0.9861
    Epoch 16/20
    18/18 [==============================] - 0s 2ms/step - loss: 0.0598 - accuracy: 0.9930
    Epoch 17/20
    18/18 [==============================] - 0s 3ms/step - loss: 0.0499 - accuracy: 0.9913
    Epoch 18/20
    18/18 [==============================] - 0s 3ms/step - loss: 0.0524 - accuracy: 0.9913
    Epoch 19/20
    18/18 [==============================] - 0s 2ms/step - loss: 0.0362 - accuracy: 0.9948
    Epoch 20/20
    18/18 [==============================] - 0s 2ms/step - loss: 0.0330 - accuracy: 0.9948





    <tensorflow.python.keras.callbacks.History at 0x7feae83c75d0>



# 3. test

---
5/5 - 1s - loss: 0.3189 - accuracy: 0.9097   
test_loss: 0.3188803493976593    
test_accuracy: 0.9096774458885193   



```python
# model을 학습시키는 코드를 직접 작성해 보세요.
# Hint! model.evaluate()을 사용해 봅시다.
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print("test_loss: {} ".format(test_loss))
print("test_accuracy: {}".format(test_accuracy))
```

    5/5 - 1s - loss: 0.3189 - accuracy: 0.9097
    test_loss: 0.3188803493976593 
    test_accuracy: 0.9096774458885193



```python

```

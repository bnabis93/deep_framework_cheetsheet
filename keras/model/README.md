# About Keras models

Keras 공식문서를 나름대로 정리하였다.   
Keras는 model을 구성할 때, Sequential / Functional API 두가지 방법으로 구성이 가능하다.  
일반적으로 Sequential은 직관적이고 단순하지만, 복잡한 모델에 적용하기 어렵고 Functional API는 Sequential에 비해 복잡하고 어렵지만, 좀 더 복잡한 모델과 Custom model을 구성하기 용이하다.  (Multi-output, shared layer 등)

- Sequential   
- Functional API  



## Sequential 






## Functional API

Keras functional API는 좀 더 복잡하고, Customize 된 model을 구성하기 위하여 사용한다.   
다음은 Keras 공식 문서의 Functional API 코드이다. 단순히 layer를 call하는것만으로 사용이 가능하고, 모델을 호출하면 해당 모델의 Architecture를 사용하는 것 뿐만 아니라, weight도 재사용 할 수 있다.  
즉, 호출하고나면, 해당 Model이 계속 남아있고 이를 재사용하기 편하다 라고 말 할 수 있다.  

```python
from keras.layers import Input, Dense
from keras.models import Model

# This returns a tensor
inputs = Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
output_1 = Dense(64, activation='relu')(inputs)
output_2 = Dense(64, activation='relu')(output_1)
predictions = Dense(10, activation='softmax')(output_2)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels)  # starts training
```

### Methods (functional API)

functional API에서는 입력과 출력이 존재하면, model을 구성 할 수있다.  
아래의 예시를 보면 직관적으로 이해가 가능 할 것이라 생각한다.   
해당 model은 'a'가 주어졌을때를 가정하고 계산을 하게 될 것

```python
from keras.models import Model
from keras.layers import Input, Dense

a = Input(shape=(32,))
b = Dense(32)(a)
model = Model(inputs=a, outputs=b)
```

#### fit
  fit(x=**None**, y=**None**, batch_size=**None**, epochs=1, verbose=1, callbacks=**None**, validation_split=0.0, validation_data=**None**, shuffle=**True**, class_weight=**None**, sample_weight=**None**, initial_epoch=0, steps_per_epoch=**None**, validation_steps=**None**, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=**False**)  
  Model을 명시한 epoch만큼 학습을 시킨다.  

##### Arguments

- x : input data. numpy 혹은 array list 형태여야 함
- y : label data, numpy





## Keras models

sequential, functioanl API 두 방법모두 공통저긍로 가지고 있는 mothod, attributes 들이 존재한다.    

- **model.layers** : 모델을 구성하는 층들이 저장된 1차원 리스트  
- **model.inputs** : 모델 입력 텐서들이 저장된 1차원 리스트  
- **model.outputs** : 모델 출력 텐서들이 저장된 1차원 리스트  
- **model.summary( )** : 모델의 구조를 요약함  
- **model.get_config()**  : 모델의 설정이 저장된 딕셔너리를 반환함  
- **model.get_weights( )** : 모델의 weight tensor들이 **numpy 배열**로 저장된 1차원 리스트  
- **model.set_weights( )** : 모델의 weight 값을 numpy 배별의 리스트로부터 설정함. 이때 리스트에 있는 배열들의 크기는 get_weights( ) 로부터 반환된 것과 동일해야 함
- **model.save_weigths( filepath )** : 모델의 weight를 hdf5 format으로 저장
- **model.load_weights( filepath, by_name = False )** : 모델의 weight를 불러온다.(hdf5 format) 주로 학습이 완료된 model을 불러오기 위하여 사용한다. by_name = False는 모델 가중치 파일과 네트워크 구조가 동일하다 가정한다. 구조가 다르면 by_name = True를 통하여 같은 이름을 가진 층의 weight만 불러올수도 있다.
- **model.to_json( ) , model.to_yaml( )** 



### Model subclass

이를 사용하여 fully customized된 model을 구성 할 수 있다. (model을 상속받아 나만의 model 구성)    
다음은 Model을 상속받아 하위클래스로 만들어진 MLP 예제  
init에 model (layer등)이 정의되어 있고 call 부분에 forward pass가 정의되어 있는것을 볼 수 있다. 

```python
import keras

class SimpleMLP(keras.Model): #Keras.Model 상속

    def __init__(self, use_bn=False, use_dp=False, num_classes=10):
      #model layer 정의 
        super(SimpleMLP, self).__init__(name='mlp')
        self.use_bn = use_bn
        self.use_dp = use_dp
        self.num_classes = num_classes

        self.dense1 = keras.layers.Dense(32, activation='relu')
        self.dense2 = keras.layers.Dense(num_classes, activation='softmax')
        if self.use_dp:
            self.dp = keras.layers.Dropout(0.5)
        if self.use_bn:
            self.bn = keras.layers.BatchNormalization(axis=-1)

    def call(self, inputs): #forward pass
        x = self.dense1(inputs)
        if self.use_dp:
            x = self.dp(x)
        if self.use_bn:
            x = self.bn(x)
        return self.dense2(x)

model = SimpleMLP()
model.compile(...)
model.fit(...)
```


# Layers

keras layer 공식 문서 및 번역 문서를 참고하여 구성함  

- About keras layers
- Core layers
- Convolutional layers



## About keras layers





## Core layers

#### Input

```python
keras.engine.input_layer.Input()
```

케라스 Tensor를 생성함.  
케라스 Tensor는 Backend (Tensorflow, Theano 등) 에서 사용되는 Tensor에 몇가지 속성을 추가한 형태이고, 이를 통하여 간단하게 모델을 생성 할 수 있다.  
예를 들어 a,b,c가 keras tensor라 하면, 다음과 같은 model을 구성가능하다.  
일반적으로 model을 구성하는 함수에 Input 함수를 넣고, 외부에서 fit을 통하여 data를 feeding 한다.  

```python
model = Model(input =[a,b], output = c)
```

keras tensor에는 다음과 같은 속성이 추가된다.   

- _keras_shape : integer 형태의 tuple, keras model에 입력을 feeding하기 위하여 shape를 알려주는 역할로 이해함
- _keras_history : 텐서에 적용되는 마지막 층. 전체 모델의 그래프를 생성 할 수 있음    

##### Arguments

- **shape** : integer tuple 형태, input의 shape를 결정한다. 

  

#### Permute

주어진 패턴에 따라 입력의 차원을 치환한다. (바꾼다)  

##### Example

```python
model = Sequential()
model.add(Permute((2, 1), input_shape=(10, 64)))
# now: model.output_shape == (None, 64, 10)
# note: `None` is the batch dimension
# 차원이 (10,64)에서 (None, 64, 10)으로 바뀜 이는 sample부분은 무시하기때문
```


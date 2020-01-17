# Keras Backend

Keras는 low-level의 연산을 제공하지는 않는다. Tensorflow, Theano, CNTK를 Backend로 사용한다.   
우리는 Keras 구현을 특정 라이브러리 (위에서 언급한 3개의 Backend를 처리하기 위한) 에 묶어 버리고, 특정한 backend 엔진을 사용함으로써 low-level의 연산을 할 수 있게 된다.  
정리하자면 Keras는 high-level framework이므로 직접적인 low-level 연산 (Tensor 곱, Tensor 합 등) 은 할 수가 없다. 
이 말은 즉, Tensorflow, Theano 등의 low-level framework를 좀 더 편리하게 이용하기 위한 lib라고 할 수 있다.  
따라서 low-level 연산을 이용하기 위해서는 backend에 접근해야 하는데, 이를 하기 위한 model-level의 lib이다.  



### Sum

```python
kears.backend.sum(x, axis = None, keepdims = False)
```

지정된 축에 다른 텐서 값들의 합

#### Arguments

- **x** : Tensor or Variable

- **axis** : [ -rank(x), rank(x) ] ㅓㅁ위의 integers 튜플 or 축, None이면 모든 차원에 대한 합  
  **axis example**  

  ```python
  a = np.arange(150).reshape((2, 3, 5, 5))
  print(np.shape(a)) #(2, 3, 5, 5)
  
  b = a.sum(axis = 0)
  print(np.shape(b)) #(3, 5, 5)
  
  c = a.sum(axis=0, keepdims=True)
  print(np.shape(c)) #(1, 3, 5, 5)
  
  d = a.sum(axis=(0,1))
  print(np.shape(d)) # (5, 5)
  
  e = a.sum(axis = 1, keepdims=True)
  print(np.shape(e)) #(2, 1, 5, 5)
  
  f = a.sum(axis = (0,2,3))
  print(np.shape(f)) #(3,)
  
  # 어느 방향으로 합을 구할 것인가
  
  #https://stackoverflow.com/questions/54126451/what-does-axis-1-2-3-mean-in-k-sum-in-keras-backend
  ```

  

- **keepdims** : boolean, 차원이 유지되고 있는가?

#### Numpy implementation

```python
def sum(x, axis = None, keepdims = False):
	if isinstance(axis, list):
    axis = tuple(axis)
	return np.sum(x, axis = axis, keepdims = keepdims)
```


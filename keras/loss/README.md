## Loss function

- Semantic segmentation
- keras custom loss function



### Semantic segmentation loss

----

Semantic segmentation 에 사용되는 loss들

#### Cross Entropy Loss

$Cross\ Entropy\ Loss = {-\sum{p(y)\log{q(y)}}}$   /  y = label, p = data 원분포, q = model이 예측한 분포  
두 확률분포 'p'와 'q'의 차이를 계산하는 함수. Cross Entropy를 최소화 함으로써 'q'의 분포를 'p'에 근사한다. 

-  #### Binary cross entropy  
$BCE = -{1\over N}\sum({y\log {y'}+ (1-y)\log{1-y'}})$     / y = label, y' = model prediction
  ```python
  # Reference : https://github.com/keras-team/keras/blob/master/keras/losses.py
  # Keras github
  # 더 깊이 들어가려면 tensorflow code를 뜯어봐야함.
  
  def binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0):
    #y_true => true label / y_pred => model predict
      y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
      y_true = K.cast(y_true, y_pred.dtype)
      if label_smoothing is not 0:
          smoothing = K.cast_to_floatx(label_smoothing)
          y_true = K.switch(K.greater(smoothing, 0),
                            lambda: y_true * (1.0 - smoothing) + 0.5 * smoothing,
                            lambda: y_true)
      return K.mean(
          K.binary_crossentropy(y_true, y_pred, from_logits=from_logits), axis=-1)
    
    
  # Tensorflow implementation
  def binary_crossentropy(target, output, from_logits=False):
    if not from_logits:
      if (isinstance(output, (ops.EagerTensor, variables_module.Variable)) or
          output.op.type != 'Sigmoid'):
        epsilon_ = _constant_to_tensor(epsilon(), output.dtype.base_dtype)
        output = clip_ops.clip_by_value(output, epsilon_, 1. - epsilon_)
  
        # Compute cross entropy from probabilities.
        bce = target * math_ops.log(output + epsilon())
        bce += (1 - target) * math_ops.log(1 - output + epsilon())
        return -bce
      else:
        # When sigmoid activation function is used for output operation, we
        # use logits from the sigmoid function directly to compute loss in order
        # to prevent collapsing zero when training.
        assert len(output.op.inputs) == 1
        output = output.op.inputs[0]
    return nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)
  ```

  

- #### Categorical cross entropy  

  ```python
  # Reference : https://github.com/keras-team/keras/blob/master/keras/losses.py
  
  def categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0):
      y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
      y_true = K.cast(y_true, y_pred.dtype)
  
      if label_smoothing is not 0:
          smoothing = K.cast_to_floatx(label_smoothing)
  
          def _smooth_labels():
              num_classes = K.cast(K.shape(y_true)[1], y_pred.dtype)
              return y_true * (1.0 - smoothing) + (smoothing / num_classes)
  
          y_true = K.switch(K.greater(smoothing, 0), _smooth_labels, lambda: y_true)
      return K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)
    
    
  # Tensorflow implementation
  def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    if not from_logits:
        if (isinstance(output, (ops.EagerTensor, variables_module.Variable)) or
            output.op.type != 'Softmax'):
          # scale preds so that the class probas of each sample sum to 1
          output = output / math_ops.reduce_sum(output, axis, True)
          # Compute cross entropy from probabilities.
          epsilon_ = _constant_to_tensor(epsilon(), output.dtype.base_dtype)
          output = clip_ops.clip_by_value(output, epsilon_, 1. - epsilon_)
          # 위의 식과 같이 구현되어있음을 확인 할 수 있음
          return -math_ops.reduce_sum(target * math_ops.log(output), axis)
        else:
          # When softmax activation function is used for output operation, we
          # use logits from the softmax function directly to compute loss in order
          # to prevent collapsing zero when training.
          # See b/117284466
          assert len(output.op.inputs) == 1
          output = output.op.inputs[0]
      return nn.softmax_cross_entropy_with_logits_v2(
          labels=target, logits=output, axis=axis)
  ```
  

##### Reference

- Cross entropy loss : [https://ratsgo.github.io/deep%20learning/2017/09/24/loss/](https://ratsgo.github.io/deep learning/2017/09/24/loss/) 



----

#### Dice Loss
Dice Coefficient is 2 \* the Area of Overlap divided by the total number of pixels in both images.  
<img src="https://miro.medium.com/max/429/1*yUd5ckecHjWZf6hGrdlwzA.png" alt="img" style="zoom:50%;" />   
Can obtain binary output.  
It it very similar to IoU. (Reference, Dice vs IoU)

  ```python
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
  ```

##### Reference

- Dice coefficient, loss : https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
- Dice vs IoU : https://stats.stackexchange.com/questions/273537/f1-dice-score-vs-iou/276144#276144

----

#### Cross entropy + Dice



#### Jaccard Distance Loss (IoU)



#### Focal Loss



----

### Kears custom loss function

1. default loss input
2. loss 계산시 axis에 대하여
3. example



#### Default loss input (y_true, y_pred)

loss function에 Tensorflow / Theano의 symbolic function을 parameter로 전달 할 수 있다. symbolic function은 다음의 두 parameters 를 전달받아 각 데이터 포인트에 대한 스칼라 값을 반환한다.

- **y_true**  :  정답 label, Tensorflow / Theano의 tensor  
  batch를 포함한다. 따라서 **shape의 첫번째 차원은 batch size**와 같다.(batch가 1이더라도)     
- **y_pred** :  model의 예측값, y_true와 shape가 같고, 역시나 Tensorflow / Theano의 tensor이다.  

  

#### loss 계산시 axis에 대하여

내가 궁금했던것은 다음과 같다.  
 Model을 구성하고 이에대한 input / output shape를 출력하면 (None, shape 1, shape2) 형식으로 출력된다. **'None' 부분은 데이터의 크기**에 관한 부분으로 알고 있다. model을 구성하고 끝부분에 loss function을 계산하는 함수를 넣어주는데 이때 loss function의 차원에 관하여 알고싶었다. **첫번째 차원이 batch size인지, 혹은 바로 data의 shape가 되는지 (batch 고려 x)**    
**=> loss의 첫번째 차원은 batch size이다.** 

  

#### Example







##### Reference

- https://stackoverflow.com/questions/46663013/what-is-y-true-and-y-pred-when-creating-a-custom-metric-in-keras








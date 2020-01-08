### Keras Data generator Manual

- Class 설명
- Flow?̊̈ Flow from dir?
- 그래서 어떻게 써야하는건데?



---

#### Class 설명

**ImageDataGenerator class**

Tensor image의 batch를 real-time으로 augmentation 하는 class

공식 example에서는 다음과 같은 흐름으로 진행한다. (val을 안써서 아쉽다)
Class 선언 -> Class.Fit (only Train) -> fit_generator 
**해당 class에서도 validation split이 가능하다 **

- **Fit function**
  - Input
    should have rank = 4, image의 ch (gray, RGB, ...) 에 따라 ch-value가 바뀐다. (gray = 1, RGB = 3)
- **Flow**
  - Input
    rank 4의 numpy array 나 tuple 형태
    First element -> image가 들어있어야함 (이미지 갯수, img[1] = 이미지)
    여튼 tensor data가 들어가면 되는 듯

**Fit_generator( ) method**
이 친구는 parameter를 보기위해서 넣어뒀다.

- Generator (first param)
  generator나 sequence instance 형태가 입력으로 들어간다.
  따라서 generator의 형태는 튜플형태의  (inputs, target) or (inputs,  targets, weights) 꼴
- Steps per epoch
  param을 update하는 step에 대한 설정 
  일반적으로 len(generator) / batch size로 함
- Validation data
  위의 generartor와 같은 형태의 data가 들어와야 함



---

#### Reference

Kears 공식 Documentation : https://keras.io/preprocessing/image/ 

Image genereator source : https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py#L233

keras fit generator : https://keras.io/models/sequential/


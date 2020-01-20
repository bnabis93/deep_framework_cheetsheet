# Issue

1. 학습을 시켰는데 loss가 변하지 않는 경우



----

### Loss가 변하지 않을때

1. Check the learning rate, lr이 너무 높은것일수도 있다. (Hyper parameter 확인)
2. input값이 알맞은 범위내에 존재하는지, noramlize는 잘 되었는지 확인
3. Bachnoramlization을 추가해본다.
4. Activation function이 잘못되었을 수도 있다.
5. Class imbalance?



#### Reference

- https://github.com/keras-team/keras/issues/2711

  


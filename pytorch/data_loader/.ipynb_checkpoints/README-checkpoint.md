## Data loader

처음 keras에서 pytorch로 넘어왔을 때, 가장 적응하기 힘들었던 부분은 단연코 Data loade 였다.  
Data를 흘려보내고 싶은데 어떤 형태로 들어가야 하는지, loader라는 class를 많이들 만들던데 이는 또 어떻게 사용하는건지  
이러한 의문들과 나름의 해답 그리고 나와 같은 사람들이 내 자료를 보고 해매지 않았으면 하느 마음으로 하나하나 만들어 보려 한다.  

#### Table of contents

- Data loader란?
- Map-style, iterable-style datasets
- 
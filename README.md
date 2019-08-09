# Keras_Stock_Price_Prediction

目的：使用類神經網路來預測股價

2019/08/09 update
原本預計用 keras，結果最後直接從 tensorflow2.0 搞，算是用了廣義的 keras 吧！？
tensorflow2.0 有點猛，汲取了 keras 的親切，還可以直接創 dataset 來處理訓練資料。
大概 80% 的時間在處理資料，真正 machine learning 的部分幾行就搞定了...

內容是利用過去60天的加權指數價格資料訓練，預測未來10天的收盤價。採用 LSTM。

結果：
![image](https://github.com/dpong/Keras_Stock_Price_Prediction/blob/master/Figure_1.png)

![image](https://github.com/dpong/Keras_Stock_Price_Prediction/blob/master/Figure_2.png)

PrIMuS-OMR
===

###### tags: `github README.md`

![](https://i.imgur.com/9GB8cw7.png)

這是一個 end-to-end 的 OMR model，輸入一張五線譜圖片，輸出 musicXML 五線譜格式

整個網路使用的是 CRNN 架構，最後再加以改進成使用 ResNet-CRNN 架構

## 使用方法

* 訓練 model 
```
python3 PrIMuS.py
```

* evaluate
```
python3 PrIMuS_Predict.py
```

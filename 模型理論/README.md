# 模型理論
## RNN、LSTM
這邊先以一個簡單、只有三層的網路為例  
輸入為 x<sub>1</sub>、x<sub>2</sub>，經過兩個隱藏層 (皆為全連接層) 後，輸出 y<sub>1</sub>、y<sub>2</sub>  
示意圖如下：  
<center>

![簡易 NN 示意圖](https://github.com/ropeshen888/BERT_Practice/blob/main/%E6%A8%A1%E5%9E%8B%E7%90%86%E8%AB%96/image_for_README/NN.png?raw=true)
</center>

然而，如果我們現在有一串輸入是有前後文關係的(如：一句話、推薦順序、語音)，一般的 NN 是沒有辦法查覺到的(每個輸入、輸出都是各自獨立的，沒辦法找到輸入前後文的相關性)。  

為了解決這樣的問題，**RNN** 就被提出來了
<center>

![簡易 RNN 示意圖](https://github.com/ropeshen888/BERT_Practice/blob/main/%E6%A8%A1%E5%9E%8B%E7%90%86%E8%AB%96/image_for_README/RNN.png?raw=true)
</center>

上圖為一個簡易的 RNN 示意圖  
比起一般 NN，RNN 多了儲存單位 a<sub>1</sub>、a<sub>2</sub>。隱藏層的內容會分別存進儲存單位存放起來，並在下一次輸入進來時，和輸入一起餵進隱藏層。  
這樣就達成了 "提取歷史資訊" 的目的
## Auto-Encoder
## Seq2seq
## BERT
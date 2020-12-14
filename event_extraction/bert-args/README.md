### 运行的环境
```
python==3.7.4
pytorch==1.3.1
pytorch-crf==0.7.2
pytorch-transformers==1.2.0
```
或
```
pip install -r requirements.txt
```

### Train
```
bash run.sh
```

### Test
修改 run.sh 中的参数 

    --do-test True

然后

```
bash run.sh
python test_mine.py
```

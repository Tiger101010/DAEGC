# DAEGC
An PyTorch implement of [Attributed Graph Clustering: A Deep Attentional Embedding Approach](https://www.ijcai.org/Proceedings/2019/0509.pdf)

**You can run the code on [Colab Notebook](https://colab.research.google.com/drive/1q2LBRiUqHgtyk2QMa3fy7kPZdbso8ilA?usp=sharing)**

## Result on Cora

|Model|ACC|NMI|F-score|ARI|
|---|---|---|---|---|
|DAEGC|0.704|0.528|0.682|0.496|
|my DAEGC|0.707|0.543|0.693|0.484|

*Didn't run kmeans 50 times to get average score for my implement*

*Just take the best score*

## Result on Citeseer

|Model|ACC|NMI|F-score|ARI|
|---|---|---|---|---|
|DAEGC|0.672|0.397|0.636|0.410|
|my DAEGC|0.687|0.417|0.643|0.441|

*Didn't run kmeans 50 times to get average score for my implement*

*Just take the best score*

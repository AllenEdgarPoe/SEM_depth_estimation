# SEM_depth_estimation
---
### Training
- First network training 
```
python GDN_main.py ./your/dataset/path --epochs 50 --batch_size 20 --gpu_num 0,1 --mode DtoD
```
- Second network training
```
python GDN_main.py ./your/dataset/path --epochs 50 --batch_size 20 --gpu_num 0,1 --mode RtoD
```

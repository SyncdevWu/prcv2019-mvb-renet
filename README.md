### SetUp
```
1. ubuntu installs anaconda3 5.X
2. build virtual envs  
   >  conda create -n bagreid python=3.6  
   >  source activate bagreid
3. conda install package
   >  conda install pytorch torchvision cudatoolkit=9.0 -c pytorch 
   >  conda install pandas
   >  pip install tensorboardX yacs
4. > git clone https://github.com/NVIDIA/apex
   > cd apex
   > pip install -v --no-cache-dir ./
```
### Data preparation
```
1. download mvb dataset from http://volumenet.cn/#/
2.
   > cd Bag_ReID
   > mkdir data 
   > mv MVB_train/Image data/MVB/bounding_box_train
   > mv MVB_train/Info/train.json data/MVB/train.json
   > mv MVB_val/Image/gallery data/MVB/gallery
   > mv MVB_val/Info/val_gallery.json data/MVB/val_gallery.json
   > mv MVB_val/Image/probe data/MVB/probe
   > mv MVB_val/Info/val_probe.json data/MVB/val_probe.json
```

### Train
```
   > python train.py
```
### Result

|            | Rank-1 | Rank-3 | Rank-5 | Rank-10 |
| ---------- | :----: | :----: | :----: | :-----: |
| 2019-07-27 | 84.32% | 94.20% | 96.67% | 98.76%  |
| 2019-08-20 | 84.98% | 95.44% | 97.24% | 98.86%  |
| 2019-08-29 | 88.69% | 96.20% | 98.38% | 99.05%  |


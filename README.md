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
2. > git clone https://github.com/wuyuejinxia/prcv2019-mvb-renet.git
   > cd prcv2019-mvb-renet
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

### Citation
```
You are encouraged to cite the following papers if this work helps your research.
@misc{zhang2019mvb,
      title={MVB: A Large-Scale Dataset for Baggage Re-Identification and Merged Siamese Networks},
      author={Zhulin Zhang and Dong Li and Jinhua Wu and Yunda Sun and Li Zhang},
      year={2019},
      eprint={1907.11366},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
     }
url: https://arxiv.org/abs/1907.11366
```



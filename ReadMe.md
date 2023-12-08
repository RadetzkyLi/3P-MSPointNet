## Pointwise predictions using 3P-MSPointNet

### Summary

This is an PyTorch implementation of our **3P-MSPointNet** semantic segmentaion network for **pointwise transportation mode identification** using GPS data. In addition, data preprocessing, prediciotn results post-processing, GPS trip visualization  and the corrected [GeoLife](https://www.microsoft.com/en-us/download/details.aspx?id=52367) dataset are also included here. 

By proposing a on-stage framework, this work aims to directly predict transportation modes of each GPS point in a trip *without* dividing the trip into signle-one mode segments. Compared to dominant two-stage methods, which divide the trip into segments with only one transportation mode first and then classify these segments, our method can leverage more contextual information and thus achieve higher overall identification accuracy. By replacing convolutions and poolings with causal convolutions and causal pooling respectively, our method can achieve **real-time prediction**. In addition, our model is light-weighted and receive trips with various lengths. 

It's recomended to open an issue for further information about the methodology. Or you can contact the author by e-mail ([lirs926535@outlook.com](lirs926535@outlook.com))

### Requirements

```python
python >= 3.7
pytorch >= 1.6.0
numpy
pickle
folium >= 0.12.1
geopy >= 2.1.0
```

### Usage

All the described data pre-processing, models and post-processing are implemented with Python programming language using PyTorch for deep learning models. Reproduced works lie in author's another project [TrajYOLO-SSD](https://github.com/RadetzkyLi/TrajYOLO-SSD)  , in which ```/reproduce/ClassicCls.py```,``` /reproduce/DeepCls.py```, ```/reproduce/ClassicSeg.py```are implementations of two-stages methods using classic classifiers, two-stage methods using deep learning algorithms and one-stage methods using classic classifiers, respectively. 

There are the following 4 folders and 2 files:

1. **data**

   * `Traj Label Each - C.rar` : The corrected GeoLife dataset whose annotations were corrected manually with the help of trajectory visualization on map.  The meanings of fields are the same as that of original GeoLife.

2. **layers**

   * Containing some costomized layers and the 3P-MSPointNet model.

3. **processing**

   * Runing `data_cleaning.py`, `pre__processing.py` and `DL_data_creation.py` in order to extract pointwise motion characteristics from raw GPS trajectories, and the processed data can be fed into neural networks directly. This part refers to another [repository](https://github.com/sinadabiri/Deep-Semi-Supervised-GPS-Transport-Mode).

   * `post_processing.py` is used to refine predictions from 3P-MSPointNet to reduce dis-continuity.

4. **utils**

   * Some common used functions and map visualization functions.

5. **test.py**

   - Given the trained model, the test set will be inferred and evaluation metrics calculated.

6. **train.py**

   - Set config and train model.

### Citation

If you find our work useful in your research, please consider citing:

```
@article{LI2023104127,
title = {A novel one-stage approach for pointwise transportation mode identification inspired by point cloud processing},
journal = {Transportation Research Part C: Emerging Technologies},
volume = {152},
pages = {104127},
year = {2023},
issn = {0968-090X},
doi = {https://doi.org/10.1016/j.trc.2023.104127},
url = {https://www.sciencedirect.com/science/article/pii/S0968090X2300116X},
author = {Rongsong Li and Zi Yang and Xin Pei and Yun Yue and Shaocheng Jia and Chunyang Han and Zhengbing He},
}
```

## License

Our code is released under MIT License (see LICENSE file for details).

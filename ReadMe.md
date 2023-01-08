## Pointwise predictions using 3P-MSPointNet

### summary

This is an pytorch implementation for out 3P-MSPointNet semantic segmentaion network. In addition, data preprocessing, prediciotn results post-processing, GPS trip visualization  and the corrected [GeoLife](https://www.microsoft.com/en-us/download/details.aspx?id=52367) dataset are also included here. 

By proposing a on-stage framework, this work aims to directly predict transportation modes of each GPS point in a trip without dividing the trip into signle-one mode segments. Compared to dominant two-stage methods, which divide the trip into segments with only one transportation mode first and then classify these segments, our method can leverage more context information and thus achieve higher overall identification accuracy. By replacing convolutiions and poolings with causal convolutions and pooling respectively, our method can achieve real-time prediction. In addition, our model is light-weighted and receive trips with various lengths. 

If you find our work useful in your research, please consider citing:

**Li, R., Yang, Z., Pei, X., Yue, Y., Jia, S., Han, C. and He, Z., 2022. A One-Stage Framework for Point-Based Transportation Mode Identification Using Gps Data. *Available at SSRN 4158243*.**

which is available at: http://ssrn.com/abstract=4158243. 

It's recomended to open an issue for further information about the methodology. Or you can contact the author by e-mail ([lirs926535@outlook.com](lirs926535@outlook.com))

### requirements

```python
python >= 3.7
pytorch >= 1.6.0
numpy
pickle
folium >= 0.12.1
geopy >= 2.1.0
```

### usage

All the described data pre-processing, models and post-processing are implemented with Python programming language using PyTorch for deep learning models. Reproduced works lie in author's another project ``TrajYOLO-SSD``  , in which ```/reproduce/ClassicCls.py```,``` /reproduce/DeepCls.py```, ```/reproduce/ClassicSeg.py```are implementations of two-stages methods using classic classifiers, two-stage methods using deep learning algorithms and one-stage methods using classic classifiers. 

Their are the following 4 folders:

1. **data**

   * `Traj Label Each - C.rar` : The corrected GeoLife dataset whose annotations were corrected manually with the help of trajectory visualization on map. 

2. **layers**

   * Containing some costomized layers and the 3P-MSPointNet model.

3. **processing**

   * `pre__processing.py` and `DL_data_creation.py` are utilized to extract pointwise motion characteristics from raw GPS trajectories.

   * `post_processing.py` is used to refine predictions from 3P-MSPointNet to reduce dis-continuity.

4. **utils**

   * Some common used functions and map visualization function.

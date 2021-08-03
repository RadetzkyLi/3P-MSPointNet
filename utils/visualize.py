#!/usr/bin/env python
# coding: utf-8


import map_util
import folium
from folium import plugins
import math
from imp import reload
import os
import time


# 1. Basic functions for visulization.

## 1.1 transform GPS orrdinates
'''
    The original GPS trajetory data is in WGS-84 coordinate systerm.
    When we use different map tiles, it's necessary to do coordinate transformation
    because diffrent coordinate systerms are used under various map providers, such as
    OpenStreetMap uses WGS-84, Gaode and Tencent Map use GC-J02, Baidu Map uses BD09.
    If coordinate systerms between GPS data and selected map provider are inconsistent,
    shifting occurs, i.e., the visualized position drifts off the actual recorded position.
    
    The optional map providers include 'OpenStreetMap','GaodeStreet','GaodeImage', 'Tencent'
        and 'Baidu'. It's faster to load map in China when using the last four.
'''
# exchange Chinese
def parse_zhch(s):
    return str(str(s).encode('ascii' , 'xmlcharrefreplace'))[2:-1]

def modes2str(modes):
    '''
    Transform modes list to string.
    : param modes:transpotation modes of list,such as ['subway','bus'];
    : return: string of modes with upper first letter,such as 'SubwayBus'.
    '''
    string = ""
    for mode in modes:
        string = string + mode.capitalize()
    return string
    
def identical_transform(point):
    return point
def transform_coord(map_name = 'OpenStreetMap'):
    if map_name == 'OpenStreetMap':
        return identical_transform
    elif map_name == 'GaodeStreet' or map_name == 'GaodeImage' or \
         map_name == 'Tencent':
        return map_util.wgs84_to_gcj02
    elif map_name == 'Baidu':
        return map_util.wgs84_to_bd09
    else:
        raise ValueError('Invalid map name:',map_name)


## 1.2 Visulization 

Mode2Index = {"walk":0,
              "run":8,
              "bike":1,
              "bus":2,
              "car":3,
              "taxi":4,
              "subway":5,
              "railway":6,
              "train":7,
              "motocycle":8,
              "boat":8,
              "airline":8,
              "other":8}
Index2Color = {0:'red',1:'beige',2:'blue',3:'green',4:'orange',5:'pink',6:'purple',7:'gray',8:'white'}
'''
Display the GPS trajactory as density map, which is saved as a html and opened by browsers.
:param traj_pts : trajectory data, [N,3], each point is denoted by (latitude,longitude,mode)
                  , whose data structure is (float,float,int).
:param outputdir : directory to output;
:param file_name : file name of the html;
:param map_name : map prodivers, one of 'OpenStreetMap', 'GaodeStreet', 'GaodeImage', 'Tencent'
        and 'Baidu'.
'''
def visualize_traj_density(traj_pts,
                           output_dir='',
                           file_name = 'traj_visual',
                          map_name = 'OpenStreetMap'
                          ):
    tiles = map_util.Map_Tiles[map_name]
    attr = map_util.Map_Attr[map_name]
    transformer = transform_coord(map_name)
    my_map = folium.Map(location = traj_pts[0][0:2],zoom_start = 12,
                       tiles = tiles ,
                       attr = attr)
    marker_cluster = plugins.MarkerCluster()
    for point in traj_pts:
        folium.Marker(transformer(point[0:2]) ,
                      icon = folium.Icon(color = Index2Color[point[3]])
                     ).add_to(marker_cluster)
    marker_cluster.add_to(my_map)
    my_map.add_child(folium.LatLngPopup())
    my_map.save(output_dir + file_name + ".html")
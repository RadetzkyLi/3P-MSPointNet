#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math



# # 1. Some Constant

# In[ ]:


# some constant about map
Map_Tiles = {'GaodeStreet':'http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}',
                'GaodeImage':'http://webst02.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}',
               'Tencent':'http://rt{s}.map.gtimg.com/realtimerender?z={z}&x={x}&y={y}&type=vector&style=0',
            'OpenStreetMap':'OpenStreetMap'}
Map_Attr = {'OpenStreetMap':'default',
            'GaodeStreet':'&copy; <a href="http://ditu.amap.com/">高德地图</a>',
            'GaodeImage':'&copy; <a href="http://ditu.amap.com/">高德地图</a>',
           'Tencent':'&copy; <a href="http://map.qq.com/">腾讯地图</a>'}


# In[4]:


x_pi = 3.14159265358979324 * 3000.0 / 180.0
pi = 3.1415926535897932384626
a = 6378245.0
ee = 0.00669342162296594323


# # 2. Coordinate Transformation 
# among Dirrerent Coordinate System.
# Reference to 'https://segmentfault.com/a/1190000017714618 '

# In[13]:


def gcj02_to_bd09(location):
    '''
    Transform point from GCJ-02 to Bd-09
    :param location:[lat,lon]
    '''
    lat,lon = location[0],location[1]
    z = math.sqrt(lon * lon + lat * lat) + 0.00002 * math.sin(lat * x_pi)
    theta = math.atan2(lat,lon) + 0.000003 * math.cos(lon * x_pi)
    bd_lon = z * math.cos(theta) + 0.0065
    bd_lat = z * math.sin(theta) + 0.006
    return [bd_lat,bd_lon]

def bd09_to_gcj02(location):
    '''
    Transform point from BD-09 to GCJ-02
    :param location:[lat,lon]
    '''
    bd_lat,bd_lon = location[0],location[1]
    x = bd_lon - 0.0065
    y = bd_lat - 0.006
    z = math.sqrt(x*x + y*y) - 0.00002 * math.sin(y * x_pi)
    theta = math.atan2(y,x) - 0.000003 * math.cos(x * x_pi)
    gg_lon = z * math.cos(theta)
    gg_lat = z * math.sin(theta)
    return [gg_lat,gg_lon]


# In[21]:


def wgs84_to_gcj02(location):
    '''
    Transform point from WGS84 to GCJ02
    :param:location:[lat,lon]
    '''
    lat,lon = location[0],location[1]
    if out_of_china(lat,lon):
        return [lat,lon]
    d_lat = _transformlat(lat - 35.0,lon - 105.0)
    d_lon = _transformlng(lat - 35.0,lon - 105.0)
    rad_lat = lat / 180.0 * pi
    magic = math.sin(rad_lat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    d_lat = (d_lat * 180.0) / ((a * (1-ee))/(magic * sqrtmagic) * pi)
    d_lon = (d_lon * 180.0) / (a / sqrtmagic * math.cos(rad_lat) * pi)
    mglat = lat + d_lat
    mglon = lon + d_lon
    return [mglat,mglon]

def gcj02_to_wgs84(location):
    """
    GCJ02 to WGS84
    :param location:[lat,lon]
    :return:
    """
    lat,lng = location[0],location[1]
    if out_of_china(lat,lng):
        return lng, lat
    dlat = _transformlat(lat - 35.0,lng - 105.0)
    dlng = _transformlng(lat - 35.0,lng - 105.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [lat * 2 - mglat,lng * 2 - mglng]

def _transformlat(lat,lng):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat +           0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * pi) + 40.0 *
            math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * pi) + 320 *
            math.sin(lat * pi / 30.0)) * 2.0 / 3.0
    return ret

def _transformlng(lat, lng):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng +           0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * pi) + 40.0 *
            math.sin(lng / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * pi) + 300.0 *
            math.sin(lng / 30.0 * pi)) * 2.0 / 3.0
    return ret

def out_of_china(lat,lng):
    """
    Judge if in China.If not,don't offset
    :param lat:
    :param lng:
    :return:
    """
    return not (73.66 < lng < 135.05 and lat > 3.86 and lat < 53.55)


# In[24]:


def bd09_to_wgs84(location):
    bd_lat,bd_lon = location[0],location[1]
    [lat,lon] = bd09_to_gcj02([bd_lat,bd_lon])
    return gcj02_to_wgs84([lat,lon])


def wgs84_to_bd09(location):
    lat,lon = location[0],location[1]
    [lat,lon] = wgs84_to_gcj02([lat,lon])
    return gcj02_to_bd09([lat,lon])


# In[18]:


def baidu_to_google(location):
    lat,lng = location[0],location[1]
    result5 = bd09_to_wgs84([float(lat),float(lng)])
    return result5


def google_to_baidu(location):
    lat,lng = location[0],location[1]
    result5 = wgs84_to_bd09([float(lat),float(lng)])
    return result5


# In[25]:


if __name__ == '__main__':
    # oribial GPS signal
    point =  [39.90777939179489,116.39123110137868]
    point_gcj = wgs84_to_gcj02(point)
    point_bd = wgs84_to_bd09(point)
    print(point_gcj)
    print(gcj02_to_wgs84(point_gcj))
    print(point_bd)
    print(bd09_to_wgs84(point_bd))


# In[ ]:





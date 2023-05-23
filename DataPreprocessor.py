# Title     : Data Preprocessor
# Objective : Preprocess the data: map the taxi coordinates into the urban area.
# Created by: Wu Shangbin
# Created on: 2021/8/3
# Required  : 
import shapely.geometry
from shapely.geometry import Point
import geopandas
import pandas as pd
import copy
import time
import numpy as np
import os

"""
提示：应该注意区域的编码问题，是从0开始还是从1开始编码.
Note: Pay attention to the coding of the urban region (starting from 0 or starting from 1)."""


def get_OD(od_str):
    """
    Extract the Origin and Destination from the OD represented by strings.
    把字符串表示的OD，提取出O和D

    :param od_str: "1,2"  (str)
    :return: 1,2  (int)
    """
    o_id, d_id = od_str.split(",")
    return int(o_id), int(d_id)


def str_to_point_shape(point_str):
    """
    Convert the longitude and latitude data (str) in the Crime data to points in shapely.geometry format
    将Crime数据中的经纬度数据（str）转化成shapely.geometry格式的点
    :param point_str: POINT (-73.91079962899994 40.67828631700008)
    :return: shapely.geometry.Point(-73.91079962899994 40.67828631700008)
    """
    point_str = point_str.strip()
    geom_type, geom_point = point_str.split(" ", 1)  
    assert geom_type == 'POINT', "Illegal Parameter, the point_str is not a Point"
    assert geom_point[:1] == "("
    assert geom_point[-1:] == ")"
    geom_point = geom_point[1:-1]  
    point_x, point_y = geom_point.split(" ")  
    point_x, point_y = float(point_x), float(point_y) 
    point = shapely.geometry.Point(point_x, point_y)
    return point


def get_project_path():
    return "./"


class DataPreprocessor:
    def __init__(self):
        self.NUM_REGION = 180
        self.taxi_dir = get_project_path() + "Data/raw_data/"
        self.data_ready_dir = get_project_path() + "Data/data_ready/"
        self.shp_path = get_project_path() + "Data/mh-180/mh-180.shp"
        self.taxi_zones = self.get_taxi_zones()

        self.find_count = 0
        self.find_mul_count = 0
        self.find_not_count = 0

    def init_find_count(self):
        self.find_count = 0      # How many points have found a corresponding area
        self.find_mul_count = 0  # How many points found multiple areas
        self.find_not_count = 0  # How many points have no corresponding area

    def get_taxi_zones(self):
        """
        Read file(.shp) for taxi zone.
        :return: taxi_zone_shape：dic zone_index -> shapely.geometry.polygon.Polygon
        """
        shp_df = geopandas.GeoDataFrame.from_file(self.shp_path)
        taxi_zone_shape = {}
        for i in range(len(shp_df)):
            # Note: The current region_ID is not 0-179, but 1-180
            taxi_zone_shape[shp_df.iloc[i]['region_id']+1] = shp_df.iloc[i]['geometry']
        print("Taxi Zone Number: ", len(taxi_zone_shape))
        return taxi_zone_shape

    def point_in_which_taxi_zone(self, point_shape):
        """
        Calculate which Taxi Zone this point is in.
        If it does not belong to any zone, - 1 is returned.
        计算一个点在哪个taxi zone里

        :param point_shape: shapely.geometry.Point
        :return taxi_zone:
        """
        count = 0
        taxi_zone = -1
        for obj_id in self.taxi_zones:
            if self.taxi_zones[obj_id].intersects(point_shape):
                taxi_zone = obj_id
                count = count + 1
        if count == 0:
            self.find_not_count = self.find_not_count + 1
        elif count == 1:
            self.find_count = self.find_count + 1
        else:
            self.find_mul_count = self.find_mul_count + 1
        return taxi_zone

    def concat_yellow_and_green(self, yellow_taxi_filename, green_taxi_filename, rename_dir=None):
        if rename_dir is None:
            # The column names may be different in green_tripdata and yellow_tripdata, and need to be unified
            rename_dir = {'lpep_pickup_datetime': 'tpep_pickup_datetime',
                          'Lpep_dropoff_datetime': 'tpep_dropoff_datetime',
                          'Pickup_longitude': 'pickup_longitude',
                          'Pickup_latitude': 'pickup_latitude',
                          'Dropoff_longitude': 'dropoff_longitude',
                          'Dropoff_latitude': 'dropoff_latitude'}
        yellow_file = self.taxi_dir + yellow_taxi_filename
        green_file = self.taxi_dir + green_taxi_filename
        yellow_taxi = pd.read_csv(yellow_file, low_memory=False)
        green_taxi = pd.read_csv(green_file, low_memory=False)
        green_taxi.rename(columns=rename_dir, inplace=True)
        yellow_taxi['taxi_type'] = 'yellow_taxi'
        green_taxi['taxi_type'] = 'green_taxi'
        all_taxi = pd.concat([yellow_taxi, green_taxi], sort=False)
        print("shape of all taxi record: ", all_taxi.shape)
        return all_taxi

    def add_taxi_location_zone(self, yearmonth="2015-07"):
        """
        It is applicable to the Taxi record files before 2016.
        Only files before 2016 contain accurate longitude and latitude coordinates.

        add ['PULocationID', 'DOLocationID'] to output .csv files

        input:  Data/raw_data/yellow_tripdata_2015-07.csv,  Data/raw_data/green_tripdata_2015-07.csv
        output: Data/data_ready/all_tripdata_2015-07.csv,   Data/data_ready/record_with_zone_2015-07/*.csv

        :param yearmonth:
        :return:
        """

        yellow_filename = "yellow_tripdata_" + yearmonth + ".csv"
        green_filename = "green_tripdata_" + yearmonth + ".csv"
        all_taxi = self.concat_yellow_and_green(yellow_filename, green_filename)
        need_col_names = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'pickup_longitude', 'pickup_latitude',
                          'dropoff_longitude', 'dropoff_latitude']
        all_taxi = all_taxi[need_col_names]
        all_taxi_filename = self.data_ready_dir + "all_tripdata_" + yearmonth + ".csv"
        all_taxi.to_csv(all_taxi_filename)

        record_with_zone_dir = self.data_ready_dir + "record_with_zoneID_" + yearmonth + "/"
        os.mkdir(record_with_zone_dir)

        def prepare_taxi_location_zone_begin_end(begin_index, end_index):
            t1 = time.time()
            all_taxi_need = pd.read_csv(all_taxi_filename)
            all_taxi_need = all_taxi_need[begin_index:end_index]
            all_taxi_need['PULocationID'] = -1
            all_taxi_need['DOLocationID'] = -1
            for i in range(end_index - begin_index):
                all_taxi_need.loc[i + begin_index, 'PULocationID'] = self.point_in_which_taxi_zone(
                    Point(all_taxi_need.iloc[i]['pickup_longitude'], all_taxi_need.iloc[i]['pickup_latitude']))
                all_taxi_need.loc[i + begin_index, 'DOLocationID'] = self.point_in_which_taxi_zone(
                    Point(all_taxi_need.iloc[i]['dropoff_longitude'], all_taxi_need.iloc[i]['dropoff_latitude']))
            all_taxi_need.to_csv(record_with_zone_dir +
                                 "taxi_sample_" + yearmonth + str(begin_index) + "-" + str(end_index) + ".csv")
            print(str(begin_index) + "-" + str(end_index) + "Finished, Time Consumed: ", time.time() - t1)

        num_of_all_taxi_records = len(all_taxi)
        self.init_find_count()
        # The complete file is too large, so it is processed in multiple parts
        for i in range(int(num_of_all_taxi_records // 100000) + 1):
            begin_i = i * 100000
            end_i = min(num_of_all_taxi_records, (i + 1) * 100000)
            prepare_taxi_location_zone_begin_end(begin_i, end_i)
        print("find_not_count：", self.find_not_count)
        print("find_mul_count：", self.find_mul_count)
        print("find_count：", self.find_count)
        self.init_find_count()
        return all_taxi


if __name__ == "__main__":
    dataPreprocessor = DataPreprocessor()
    dataPreprocessor.add_taxi_location_zone()

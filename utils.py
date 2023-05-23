import geopandas
import shapely
from shapely.geometry import Point


def get_taxi_zones_dic(shp_path):
    """
    Read file(.shp) for taxi zone.
    :return: taxi_zone_shape：dic zone_index -> shapely.geometry.polygon.Polygon
    """
    shp_df = geopandas.GeoDataFrame.from_file(shp_path)
    taxi_zone_shape = {}
    for i in range(len(shp_df)):
        # Note: The current region_ID is not 0-179
        taxi_zone_shape[shp_df.iloc[i]['region_id']+1] = shp_df.iloc[i]['geometry']
    # print("Taxi Zone Number: ", len(taxi_zone_shape))
    return taxi_zone_shape


def get_zone_id(longitude, latitude):
    taxi_zone_shape_dic = get_taxi_zones_dic("./Data/mh-180/mh-180.shp")
    find_count = 0
    for obj_id in taxi_zone_shape_dic:
        if taxi_zone_shape_dic[obj_id].intersects(Point(longitude, latitude)):
            taxi_zone = obj_id
            find_count = find_count + 1
    
    if find_count == 1:
        return taxi_zone
    else:
        return -1
    

if __name__ == '__main__':
    print(get_zone_id("-73.9911117553711", "40.72775650024414"))
    print(get_zone_id(-73.99695587158203, 40.73704147338867))
    print(geopandas.__version__)
    print(shapely.__version__)

"""
"""
from database.loader import DataLoader, shuffle_and_split
from typing import List, Union

files = [
    "uav_beijing/train.csv", "uav_boston/train.csv", "uav_london/train.csv",
    "uav_moscow/train.csv", "uav_tokyo/train.csv"
]

def get_cities(cities: Union[str,List]="all", val_ratio: float=0.10):
    """
    """
    city_list = [city.strip().lower() for city in cities.split(",")]
    supported = {"beijing", "boston", "london", "moscow", "tokyo"}
    invalid = set(city_list) - supported
    if invalid: 
        sys.exit(1)
    files = [f"uav_{city}/train.csv" for city in city_list]

    # Loader and shuffle data 
    loader = DataLoader()
    dtr, dts = shuffle_and_split(data=loader.load(files), val_ratio=val_ratio)
    return dtr, dts


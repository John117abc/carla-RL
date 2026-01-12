import carla


# 要卸载的图层列表 1级
LAYERS_TO_REMOVE_1 = [
    carla.MapLayer.Buildings,
    carla.MapLayer.Walls,
    carla.MapLayer.Foliage,
    carla.MapLayer.Decals,
    carla.MapLayer.ParkedVehicles,
    carla.MapLayer.Props,
    carla.MapLayer.Particles,
]

# 要卸载的图层列表 2级
LAYERS_TO_REMOVE_2 = [
    carla.MapLayer.Buildings,
    carla.MapLayer.Walls,
    carla.MapLayer.Props,
]

# 要卸载的图层列表 3级
LAYERS_TO_REMOVE_3 = [
    carla.MapLayer.Buildings,
    carla.MapLayer.Props
]

# 要卸载的图层列表 4级
LAYERS_TO_REMOVE_4 = [
    carla.MapLayer.Props
]

# 出生点
BIRTH_POINT = {
    "Town06_Opt": 6
}
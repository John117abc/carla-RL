import carla


# 要卸载的图层列表 1级
LAYERS_TO_REMOVE_1 = [
    carla.MapLayer.Buildings,
    carla.MapLayer.Walls,
    carla.MapLayer.Foliage,
    carla.MapLayer.Decals,
    carla.MapLayer.ParkedVehicles,
    carla.MapLayer.StreetLights,
    carla.MapLayer.Particles,
    carla.MapLayer.Props
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
    # "Town02_Opt": 6,
    "Town06_Opt":[
        # carla.Location(x=106.3154, y=237.56, z=0.6),  # 最左侧车道中心
        #                    carla.Location(x=106.3154, y=241.06, z=0.6),
        #                    carla.Location(x=30.3154, y=244.56, z=0.6),
                           carla.Location(x=20.3154, y=248.06, z=0.6),
        #                    carla.Location(x=20.3154, y=251.56, z=0.6)
    ],
    "Town05_Opt":[carla.Location(x=26.4, y=-207.6, z=0.3)]
}

# 出生角度
BIRTH_YAW = {
    "Town05_Opt":179.8,
    "Town06_Opt":0.0
}

# 终点
END_POINT = {
    "Town06_Opt": carla.Location(x = 300.3154,y = 248.0,z = 0.300000),
    "Town05_Opt": carla.Location(x=37.3, y=208.8, z=0.3)
}
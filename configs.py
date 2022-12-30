from data import HORIZONS


horizon_map_2x2all = {
    5: (0, 0),
    10: (0, 1),
    15: (1, 0),
    20: (1, 1)
}
horizon_map_2x2no5 = {
    10: (0, 0),
    15: (1, 0),
    20: (1, 1)
}
horizon_map_1x3 = {
    10: (0, 0),
    15: (0, 1),
    20: (0, 2)
}

horizon_maps = {
    "2x2 all": horizon_map_2x2all,
    "2x2 no5": horizon_map_2x2no5,
    "1x3": horizon_map_1x3,
    "10y": {10: (0, 0)},
    "20y": {20: (0, 0)},
}
dims = {
    "2x2 all": (2, 2),
    "2x2 no5": (2, 2),
    "1x3": (1, 3),
    "10y": (1, 1),
    "20y": (1, 1),
}
figsizes = {
    "2x2 all": (7, 6),
    "2x2 no5": (7, 6),
    "1x3": (9, 4),
    "10y": (5, 4),
    "20y": (5, 4),
}
btas = {
    #"2x2 all": [0.35, -0.45],
    #"2x2 no5": [0.35, -0.45],
    #"1x3": [-0.15, -0.4],
    #"10y": [1.3, 1.04],
    "2x2 all": [0.98,0],
    "2x2 no5": [0.98,0],
    "1x3": [0.85, 0],
    "10y": [1.33, 0.9],
    "20y": [1.33, 0.9],
}
supxlabels = {
    "2x2 all": 'Real rate over horizon',
    "2x2 no5": 'Real rate over horizon',
    "1x3": 'Real rate over horizon',
    "10y": '{0}y real rate'.format(10),
    "20y": '{0}y real rate'.format(20),
}
supylabels = {
    "2x2 all": 'Real GDP growth over horizon',
    "2x2 no5": 'Real GDP growth over horizon',
    "1x3": 'Real GDP growth over horizon',
    "10y": 'RGDP growth next {0}y'.format(10),
    "20y": 'RGDP growth next {0}y'.format(20),
}
suptitles = {
    "2x2 all": 'Real rate vs. future real GDP growth',
    "2x2 no5": 'Real rate vs. future real GDP growth',
    "1x3": 'Real rate vs. future real GDP growth',
    "10y": '{0}y real rate vs. RGDP growth next {0} years'.format(10),
    "20y": '{0}y real rate vs. RGDP growth next {0} years'.format(20),
}
titleweights = {
    "2x2 all": 'heavy',
    "2x2 no5": 'heavy',
    "1x3": 'heavy',
    "10y": None,
    "20y": None,
}
subplot_titless = {
    "2x2 all": True,
    "2x2 no5": True,
    "1x3": True,
    "10y": False,
    "20y": False,
}
legend_ncols = {
    "2x2 all": 5,
    "2x2 no5": 5,
    "1x3": 5,
    "10y": 1,
    "20y": 1,
}
save_names = {
    "2x2 all": "scatter_r_vs_gdp_multi_2x2_all",
    "2x2 no5": "scatter_r_vs_gdp_multi_2x2_no5",
    "1x3": "scatter_r_vs_gdp_multi_1x3",
    "10y": "scatter_r_vs_gdp_{0}".format(10),
    "20y": "scatter_r_vs_gdp_{0}".format(20),
}

config_objects = [horizon_maps, dims, figsizes, btas, supxlabels, supylabels,
                  suptitles, titleweights, subplot_titless, legend_ncols]
for horizon in HORIZONS:
    if horizon not in config_objects:
        for obj in config_objects:  # then copy 20y
            string = '{0}y'.format(horizon)
            obj[string] = obj['20y']
            horizon_maps[string] = {horizon: (0, 0)}
            supxlabels[string] = '{0}y real rate'.format(horizon)
            supylabels[string] = 'RGDP growth next {0}y'.format(horizon)
            suptitles[string] = '{0}y real rate vs. RGDP growth next {0} years'.format(horizon)
            save_names[string] = 'scatter_r_vs_gdp_{0}'.format(horizon)
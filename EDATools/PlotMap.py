# python3
# Author: Scc_hy
# Create date: 2021-10-28
# Func:  绘制地图
# Related Blog: https://blog.csdn.net/Scc_hy/article/details/120898390
# ========================================
from h3 import h3
import folium
from folium.plugins import HeatMap

def plot_map(clusters, color_dict=None, zoom_start=12):
    """
    params clusters: such as{'0x87309a401ffffff': 
        {'count': 1,
       'geom': (...)}, ...
   }
   for idx, row in draw_h3_df.iterrows():
    clusters[row['h3no_16']] = {'count' : row['dispath'], 
                   'geom': h3.h3_to_geo_boundary(h=row['h3no_16'])}
    """
    map_fig = folium.Map(location=[30.250538536358924, 120.19170930065695], zoom_start=zoom_start)
    def color_choose(cnt):
        color_list = ['#FFC1C1', '#EEB4B4', '#FF6A6A', '#EE6363', '#CD5555', '#8B3A3A']
        if cnt <= 4:
            return color_list[0]
        elif cnt <= 7:
            return color_list[1]
        elif cnt <= 11:
            return color_list[2]    
        elif cnt <= 15:
            return color_list[3]   
        elif cnt <= 20:
            return color_list[4]   
        else:
            return color_list[5]  

    
    for cluster in clusters.values():
        points = cluster['geom']
        ac_cnt = cluster['count']
        color = color_dict[cluster['color']] if color_dict else color_choose(ac_cnt)
        tooltip = f"{cluster['color']}-{ac_cnt}"  if color_dict else  f'{ac_cnt}' 
        map_fig.add_child(
        folium.vector_layers.Polygon(
            locations=points,
            tooltip=tooltip,
            fill=True,
            fill_color=color,#'#ff0000',
            fill_opacity=0.6,
            weight=2,
            opacity=0.7
        ))
    return map_fig


def quit_plot(draw_h3_df, color_dict=None, zoom_start=12):
    draw_h3_df['h3no_16'] = draw_h3_df['h3no'].map(lambda x:hex(int(x)))
    clusters = {}
    for idx, row in draw_h3_df.iterrows():
        clusters[row['h3no_16']] = {'count' : row['dispath'], 
                       'geom': h3.h3_to_geo_boundary(h=row['h3no_16'])}
    return plot_map(clusters, color_dict=color_dict, zoom_start=zoom_start), clusters
# Recorded script from Mayavi2
from numpy import array
try:
    engine = mayavi.engine
except NameError:
    from mayavi.api import Engine
    engine = Engine()
    engine.start()
if len(engine.scenes) == 0:
    engine.new_scene()
# ------------------------------------------- 
module_manager = engine.scenes[0].children[0].children[0]
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = array([1, 1])
module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([ 0.1,  0.8])
module_manager.scalar_lut_manager.scalar_bar_representation.moving = 1
module_manager.scalar_lut_manager.scalar_bar_representation.position = array([ 0.01,  0.15])
module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = array([100000, 100000])
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = array([1, 1])
module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([ 0.1,  0.8])
module_manager.scalar_lut_manager.scalar_bar_representation.moving = 0
module_manager.scalar_lut_manager.scalar_bar_representation.position = array([ 0.01,  0.15])
module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = array([100000, 100000])
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = array([1, 1])
module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([ 0.1       ,  0.79276673])
module_manager.scalar_lut_manager.scalar_bar_representation.position = array([ 0.01      ,  0.15723327])
module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = array([100000, 100000])
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = array([1, 1])
module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([ 0.1       ,  0.68245931])
module_manager.scalar_lut_manager.scalar_bar_representation.position = array([ 0.01      ,  0.26754069])
module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = array([100000, 100000])
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = array([1, 1])
module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([ 0.1       ,  0.49981917])
module_manager.scalar_lut_manager.scalar_bar_representation.position = array([ 0.01      ,  0.45018083])
module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = array([100000, 100000])
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = array([1, 1])
module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([ 0.1       ,  0.46546112])
module_manager.scalar_lut_manager.scalar_bar_representation.position = array([ 0.01      ,  0.48453888])
module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = array([100000, 100000])
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = array([1, 1])
module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([ 0.1       ,  0.40940325])
module_manager.scalar_lut_manager.scalar_bar_representation.position = array([ 0.01      ,  0.54059675])
module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = array([100000, 100000])
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = array([1, 1])
module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([ 0.1       ,  0.34068716])
module_manager.scalar_lut_manager.scalar_bar_representation.position = array([ 0.01      ,  0.60931284])
module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = array([100000, 100000])
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = array([1, 1])
module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([ 0.1       ,  0.34068716])
module_manager.scalar_lut_manager.scalar_bar_representation.moving = 1
module_manager.scalar_lut_manager.scalar_bar_representation.position = array([ 0.01      ,  0.60931284])
module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = array([100000, 100000])
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = array([1, 1])
module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([ 0.1       ,  0.34068716])
module_manager.scalar_lut_manager.scalar_bar_representation.position = array([ 0.01143062,  0.60931284])
module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = array([100000, 100000])
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = array([1, 1])
module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([ 0.1       ,  0.34068716])
module_manager.scalar_lut_manager.scalar_bar_representation.position = array([ 0.03718169,  0.60931284])
module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = array([100000, 100000])
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = array([1, 1])
module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([ 0.1       ,  0.34068716])
module_manager.scalar_lut_manager.scalar_bar_representation.position = array([ 0.05291845,  0.60931284])
module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = array([100000, 100000])
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = array([1, 1])
module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([ 0.1       ,  0.34068716])
module_manager.scalar_lut_manager.scalar_bar_representation.position = array([ 0.06436338,  0.60931284])
module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = array([100000, 100000])
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = array([1, 1])
module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([ 0.1       ,  0.34068716])
module_manager.scalar_lut_manager.scalar_bar_representation.position = array([ 0.06579399,  0.60931284])
module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = array([100000, 100000])
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = array([1, 1])
module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([ 0.1       ,  0.34068716])
module_manager.scalar_lut_manager.scalar_bar_representation.position = array([ 0.0758083 ,  0.60750452])
module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = array([100000, 100000])
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = array([1, 1])
module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([ 0.1       ,  0.34068716])
module_manager.scalar_lut_manager.scalar_bar_representation.position = array([ 0.08725322,  0.60750452])
module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = array([100000, 100000])
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = array([1, 1])
module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([ 0.1       ,  0.34068716])
module_manager.scalar_lut_manager.scalar_bar_representation.position = array([ 0.09297568,  0.60750452])
module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = array([100000, 100000])
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = array([1, 1])
module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([ 0.1       ,  0.34068716])
module_manager.scalar_lut_manager.scalar_bar_representation.position = array([ 0.09297568,  0.6056962 ])
module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = array([100000, 100000])
module_manager.scalar_lut_manager.scalar_bar_representation.minimum_size = array([1, 1])
module_manager.scalar_lut_manager.scalar_bar_representation.position2 = array([ 0.1       ,  0.34068716])
module_manager.scalar_lut_manager.scalar_bar_representation.moving = 0
module_manager.scalar_lut_manager.scalar_bar_representation.position = array([ 0.09297568,  0.6056962 ])
module_manager.scalar_lut_manager.scalar_bar_representation.maximum_size = array([100000, 100000])
scene = engine.scenes[0]
scene.scene.camera.position = [2122.0143833805705, 853.68971486704731, 247.60407778970642]
scene.scene.camera.focal_point = [1974.8575770716611, 706.53290855813771, 100.44727148079686]
scene.scene.camera.view_angle = 30.0
scene.scene.camera.view_up = [0.0, 0.0, 1.0]
scene.scene.camera.clipping_range = [143.9964028826318, 395.08132039984048]
scene.scene.camera.compute_view_plane_normal()
scene.scene.render()
axes = engine.scenes[0].children[0].children[0].children[2]
axes.axes.ranges = array([ 1907.97501596,  2027.29692918,   686.82376325,   741.08793613,
          92.7442    ,   107.74772   ])
axes.axes.position2 = array([ 0.5,  0.5])
axes.axes.bounds = array([ 1907.97497559,  2027.296875  ,   686.8237915 ,   741.08795166,
          92.74420166,   107.74771881])
axes.axes.position = array([ 0.,  0.])
axes.axes.label_format = '%.0f'
# ------------------------------------------- 
from mayavi.tools.show import show
show()

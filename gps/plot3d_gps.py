import tables as tb
import mayavi.mlab as ml

f = tb.openFile('/Users/fpaolo/data/gps/amery_gps_all.h5')
t = f.root.time[:]
x = f.root.x[:]
y = f.root.y[:]
h = f.root.h[:]
f.close()

ml.figure(1, size=(700, 600), fgcolor=(1, 1, 1), bgcolor=(0.5, 0.5, 0.5))
ml.points3d(x, y, h, t.round(1), mode='point')
#ml.plot3d(x, y, h, t.round(1), tube_radius=.3)
ml.colorbar(title='campaigns', orientation='vertical', 
            nb_labels=5, label_fmt='%.0f', nb_colors=5)
ml.outline()
ml.axes(nb_labels=4).axes.label_format = '%.0f'
ml.title('Amery GPS', size=.7)
ml.xlabel('x (km)')
ml.ylabel('y (km)')
ml.zlabel('h (m)')
ml.show()

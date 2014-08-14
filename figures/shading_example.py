import numpy as np 
import matplotlib.pyplot as plt 

def rgb_to_hsv_arr(arr): 
    """ fast rgb_to_hsv using numpy array """ 
    # adapted from Arnar Flatberg http://www.mail-archive.com/numpy-discussion@.../msg06147.html it now handles 
    # NaN properly and mimics colorsys.rgb_to_hsv output 
    arr = arr/255. 
    out = np.empty_like(arr) 
    arr_max = arr.max(-1) 
    delta = arr.ptp(-1) 
    s = delta / arr_max 
    s[delta==0] = 0 
    # red is max 
    idx = (arr[:,:,0] == arr_max) 
    out[idx, 0] = (arr[idx, 1] - arr[idx, 2]) / delta[idx] 
    # green is max 
    idx = (arr[:,:,1] == arr_max) 
    out[idx, 0] = 2. + (arr[idx, 2] - arr[idx, 0] ) / delta[idx] 
    # blue is max 
    idx = (arr[:,:,2] == arr_max) 
    out[idx, 0] = 4. + (arr[idx, 0] - arr[idx, 1] ) / delta[idx] 
    out[:,:,0] = (out[:,:,0]/6.0) % 1.0 
    out[:,:,1] = s 
    out[:,:,2] = arr_max 
    return out 

# code from colorsys module, should numpy'ify 
def hsv_to_rgb(h, s, v): 
    if s == 0.0: return v, v, v 
    i = int(h*6.0) # XXX assume int() truncates! 
    f = (h*6.0) - i 
    p = v*(1.0 - s) 
    q = v*(1.0 - s*f) 
    t = v*(1.0 - s*(1.0-f)) 
    if i%6 == 0: return v, t, p 
    if i == 1: return q, v, p 
    if i == 2: return p, v, t 
    if i == 3: return p, q, v 
    if i == 4: return t, p, v 
    if i == 5: return v, p, q 
    # Cannot get here 

def hsv_to_rgb_arr(arr): 
    # vectorize this! 
    out = np.empty(arr.shape, arr.dtype) 
    for i in range(arr.shape[0]): 
        for j in range(arr.shape[1]): 
            h,s,v = arr[i,j,0].item(),arr[i,j,1].item(),arr[i,j,2].item() 
            r,g,b = hsv_to_rgb(h,s,v) 
            out[i,j,0]=r; out[i,j,1]=g; out[i,j,2]=b 
    return out 

def illumination(idata,azdeg=315.0,altdeg=45.): 
    # convert alt, az to radians 
    az = azdeg*np.pi/180.0 
    alt = altdeg*np.pi/180.0 
    # gradient in x and y directions 
    dx, dy = np.gradient(idata) 
    slope = 0.5*np.pi - np.arctan(np.hypot(dx, dy)) 
    aspect = np.arctan2(dx, dy) 
    odata = np.sin(alt)*np.sin(slope) + np.cos(alt)*np.cos(slope) * np.cos(-az - aspect - 0.5*np.pi) 
    # rescale to interval -1,1 
    # +1 means maximum sun exposure and -1 means complete shade. 
    odata = (odata - odata.min())/(odata.max() - odata.min()) 
    odata = 2.*odata - 1. 
    return odata 

# test data 
X,Y=np.mgrid[-5:5:0.05,-5:5:0.05] 
Z=X+Y+np.sin(X**2+Y**2) 
# imagine an artificial sun placed at infinity in 
# some azimuth and elevation position illuminating our surface. The parts of 
# the surface that slope toward the sun should brighten while those sides 
# facing away should become darker; no shadows are cast as a result of 
# topographic undulations. 
intensity = illumination(Z) 
plt.figure() 
# plot original image 
im = plt.imshow(Z,cmap=plt.cm.copper) 
# convert to rgb, then rgb to hsv 
rgb = im.to_rgba(Z) 
hsv = rgb_to_hsv_arr(rgb[:,:,0:3]) 
# darken any pure color (on the cube facets) by keeping H fixed and adding black 
# and brighten it by adding white; for interior points in the cube we will add or 
# remove gray. 
hsv_min_sat = 1.0 
hsv_max_sat = 0.1 
hsv_min_val = 0.3 
hsv_max_val = 1.0 
hsv[:,:,1] = np.where(np.logical_and(np.abs(hsv[:,:,1])>1.e-10,intensity>0),\
        (1.-intensity)*hsv[:,:,1]+intensity*hsv_max_sat, hsv[:,:,1]) 
hsv[:,:,2] = np.where(intensity > 0, (1.-intensity)*hsv[:,:,1] +\
        intensity*hsv_max_val, hsv[:,:,2]) 
hsv[:,:,1] = np.where(np.logical_and(np.abs(hsv[:,:,1])>1.e-10,intensity<0),\
        (1.+intensity)*hsv[:,:,1]-intensity*hsv_max_sat, hsv[:,:,1]) 
hsv[:,:,2] = np.where(intensity < 0, (1.+intensity)*hsv[:,:,1] -\
        intensity*hsv_min_val, hsv[:,:,2]) 
hsv[:,:,1:] = np.where(hsv[:,:,1:]<0.,0,hsv[:,:,1:]) 
hsv[:,:,1:] = np.where(hsv[:,:,1:]>1.,1,hsv[:,:,1:]) 
# convert modified hsv back to rgb. 
rgb[:,:,0:3] = hsv_to_rgb_arr(hsv) 
plt.figure() 
# plot shaded image. 
plt.imshow(rgb) 
plt.show() 

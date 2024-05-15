import numpy as np
import matplotlib.pyplot as plt

def compute_shifted_indices(vecs,shape):
    origins = shape*(vecs<0)
    counter_origins = shape-origins
    origins_target = counter_origins-vecs
    counter_origins_target = origins+vecs
    min1 = np.minimum(origins,origins_target).astype(int)
    max1 = np.maximum(origins,origins_target).astype(int)
    min2 = np.minimum(counter_origins,counter_origins_target).astype(int)
    max2 = np.maximum(counter_origins,counter_origins_target).astype(int)
    return min1,max1,min2,max2

def strfn(vecs,array,orders):
    output = np.zeros((vecs.shape[0],len(orders)))
    min1,max1,min2,max2 = compute_shifted_indices(vecs,array.shape)
    for v in range(vecs.shape[0]):
        X1 = array[min1[v,0]:max1[v,0],min1[v,1]:max1[v,1],min1[v,2]:max1[v,2],min1[v,3]:max1[v,3]]
        X2 = array[min2[v,0]:max2[v,0],min2[v,1]:max2[v,1],min2[v,2]:max2[v,2],min2[v,3]:max2[v,3]]
        absdX = np.sqrt(np.sum((X1-X2)**2,axis=0))
        for i in range(len(orders)):
            output[v,i] = np.nanmean(absdX**orders[i])
    return output

###### UTILITIES TO CREATE LOG SPACED LAG VECTORS

def floor_non_inclusive(x):
    fl_x = np.floor(x)
    fl_x = fl_x - (fl_x==x)*1
    return fl_x
def get_circle_coords(lag,half_bin_width):
    possible_val1 = np.arange(-floor_non_inclusive(lag+half_bin_width),floor_non_inclusive(lag+half_bin_width)+1).astype(int)
    possible_val2_upper_squared = (lag+half_bin_width)**2-possible_val1**2
    possible_val2_upper = floor_non_inclusive(np.sqrt(possible_val2_upper_squared).round(5)).astype(int)
    possible_val2_lower_squared = np.maximum((lag-half_bin_width)**2-possible_val1**2,0)
    possible_val2_lower = np.ceil(np.sqrt(possible_val2_lower_squared).round(5)).astype(int)
    
    dis = possible_val2_upper-possible_val2_lower+1
    lengths = dis*2 - (possible_val2_lower==0)*1
    vecs = np.zeros((np.sum(lengths),2)).astype(int)
    i=0
    for v1 in range(len(possible_val1)):
        vecs[i:(i+lengths[v1]),0]=possible_val1[v1]
        vecs[i:(i+dis[v1]),1]=np.arange(possible_val2_lower[v1],possible_val2_upper[v1]+1)
        vecs[i+dis[v1]:(i+lengths[v1]),1]=-np.arange(possible_val2_lower[v1]+(possible_val2_lower[v1]==0)*1,possible_val2_upper[v1]+1)
        i+=lengths[v1]
    return vecs

def get_log_spaced_lags(max_lag,n_lags,max_half_bin_width,saveimg=False):
    img = np.zeros((max_lag*2+1,max_lag*2+1))
    lag_range = np.exp(np.linspace(0,np.log(max_lag+1),n_lags+2))
    valid_lags = np.unique(lag_range.astype(int))
    upper_lags = valid_lags[1:]
    lower_lags = valid_lags[:-1]
    lags_bin_range = np.minimum((upper_lags-lower_lags)/2,max_half_bin_width)
    lags = upper_lags-lags_bin_range 
    all_vecs = []
    n_samples = []
    for l in range(len(lags)):
        vecs = get_circle_coords(lags[l],lags_bin_range[l])
        all_vecs.append(vecs)
        n_samples.append(vecs.shape[0])
        img[vecs[:,0]+max_lag,vecs[:,1]+max_lag]=np.log(lags[l])
    img = img-np.min(img)
    img = img/np.max(img)
    cmap = plt.cm.magma
    image = cmap(img)
    if saveimg:
       plt.imsave('log_lags_image.png', image)

    return lags,np.concatenate(all_vecs)


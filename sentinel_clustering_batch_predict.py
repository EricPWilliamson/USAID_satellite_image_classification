"""
Uses a pickled MiniBatchKMeans cluster model to predict cluster labels for all pixels in a Sentinel-2 dataset. This can
  make future scripts that use these clusters faster, but overall, this takes more time as it processes every pixel. It
  is faster overall to not use this and only predict clusters as they are needed.

As usual, loading jp2 files is the primary bottleneck.
Prediction time is another bottleneck. Predicting a 1/10th batch with 100 clusters takes ~32s.
!!Try increasing batch size for faster processing.
!!Change output file dir and naming

Creating temporary npy files for faster reloading makes a big difference.
  Loading the jp2 files for a batch takes ~120s, loading npy files for a batch takes just ~28s. Writing the npy files takes ~80s.
  So, 10 batch loading time went from 20mins to 7.5mins.

When other Windows programs are not active, the load times are reduced significantly. I suspect disk read speed is a
  limiting factor here. NPY bands can be loaded in ~18s. Loading jp2s and saving the npys still takes ~205s. Other
  operations are still the same speed.

The total time for an entire set to be processed is 13 minutes, meaning the entire 111 sets should take ~24 hours.

!!!Left off on 45th set (i=44)...then left off on 9 of 67 (i=8). So, i=52...left off on 50of59 (i=49). So, i=101
"""
from satellite_head import *
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
from timeit import default_timer as timer
import pickle
import os.path

#Hardcoded path to satellite data dir:
SAT_DATA_DIR = "./USAID_africa/ESA/"
#Hardcoded satellite image size:
SAT_SIZE = 10980
#Hardcoded spectral band names:
BAND_LIST = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12']
BAND_REZS = [60,   10,    10,    10,   20,   20,   20,   10,   20,   60,   60,   20,   20]
#Hardcoded path to pickled model:
MBKM_PATH = './USAID_africa/Models/MBKM_grand_120.pkl'
#Harcoded path to pickled StandardScalers dir:
SS_DIR = './USAID_africa/StandardScalers/'
#Hardcoded number of pixels per batch:
BSIZE = 12056040 #Exactly 1/10th of pixels in 10980x10980 image

def make_npy_path(band_str, sat_data_path):
    TMP_PATH = "./tmp/"
    sat_set_name = sat_data_path.split('/')[-1]
    if len(sat_set_name)==0:
        sat_set_name = sat_data_path.split('/')[-2]
    return TMP_PATH + sat_set_name + '_' + band_str + '.npy'

def fast_load_band(band_str, sat_data_path):
    npy_path = make_npy_path(band_str, sat_data_path)
    if os.path.isfile(npy_path):
        return np.load(npy_path)
    else:
        #Load up the band from jp2 file:
        raw_img = get_band(band,SAT_DATA_DIR+folder)
        #If the image is not on the 10980x10980 grid, use bilinear interpolation to blow it up:
        if raw_img.shape[0]!=SAT_SIZE:
            raw_img = scipy.misc.imresize(raw_img,SAT_SIZE/raw_img.shape[0],interp='bilinear',mode='F').astype('float32')
        #Save the band for next run:
        np.save(npy_path,raw_img)
        return raw_img

def erase_band(band_str, sat_data_path):
    npy_path = make_npy_path(band_str,sat_data_path)
    if os.path.isfile(npy_path):
        os.remove(npy_path)
    else:
        print('Couldnt delete npy file.')

###Load the MBKM from pickle:
with open(MBKM_PATH,'rb') as pfile:
    mbkm = pickle.load(pfile)

###Predict clusters for every pixel in every data set:
#List of any folders in the SAT_DATA_DIR with the '.SAFE' extension:
sat_set_list = os.listdir(SAT_DATA_DIR)
sat_set_list = [s for s in sat_set_list if s[-5:]=='.SAFE']
sat_set_list = sat_set_list[101:] #!!!!!!!!!!!!!!!!!!!Pick up from previous
for i,folder in enumerate(sat_set_list):
    t1 = timer()
    print("Fitting dataset {} of {}: '{}'".format(i+1, len(sat_set_list), folder))
    ###Predict labels in batches:
    pixel_labels = np.zeros(SAT_SIZE**2).astype('uint16')
    for pix_idx in range(0,SAT_SIZE**2,BSIZE):
        spectral_pixels = np.zeros((BSIZE,len(BAND_LIST))).astype('float32')
        load_time = 0
        for b,band in enumerate(BAND_LIST):
            print(band,end='|',flush=True)
            t2 = timer()
            band_img = fast_load_band(band, SAT_DATA_DIR+folder)
            load_time += timer()-t2
            #Grab the pixels belonging to this batch:
            band_img = band_img.reshape(-1,1)[pix_idx:pix_idx+BSIZE]
            #Apply StandardScaler from the grand fit:
            with open(SS_DIR+band+'_grand_SS.pkl','rb') as pfile:
                ss = pickle.load(pfile)
            band_img = ss.transform(band_img.astype('float32'))
            #Put this band into our matrix:
            spectral_pixels[:,b] = band_img.reshape(-1)
        print("  --Bands loaded in {:.1f}s".format(load_time),end='',flush=True)

        #Predict clusters with model:
        t2 = timer()
        pixel_labels[pix_idx:pix_idx+BSIZE] = mbkm.predict(spectral_pixels)
        print("  --Clusters predicted in {:.1f}s.".format(timer()-t2))

    #Erase temp band files:
    t2 = timer()
    for band in BAND_LIST:
        erase_band(band, SAT_DATA_DIR+folder)
    print(' Bands erased in {:.2f}s'.format(timer()-t2))

    #Save cluster predictions:
    n_clusters = int(np.max(pixel_labels)+1)
    out_path = make_npy_path('{}clusters'.format(n_clusters), SAT_DATA_DIR+folder) #!!!change
    if n_clusters < 256:
        np.save(out_path,pixel_labels.reshape((SAT_SIZE,SAT_SIZE)).astype('uint8'))
    else:
        np.save(out_path,pixel_labels.reshape((SAT_SIZE,SAT_SIZE)))

    print('Set processed in {:.1f} mins'.format((timer()-t1)/60))

# ###Plot results:
# #Plot cluster labels:
# print('Plotting clusters...')
# np.random.seed(285)
# colors = np.random.rand(n_clusters,3).astype('float32')
# c = colors[pixel_labels]
# imshow_to_file(c.reshape((SAT_SIZE,SAT_SIZE,3)),
#                'grand_clustering_experiment_MBKM_clusters.png',
#                title='Grand MBKM model ({} clusters)'.format(n_clusters))
# print('Plotting TCI...')
# tci = get_band('TCI',SAT_DATA_DIR+folder)
# imshow_to_file(tci,
#                'grand_clustering_experiment_TCI.png',
#                title='TCI img (cropped)')

myalert()
print('Done')

"""
Use the partial_fit method to gradually build a MiniBatchKMeans cluster model from a very large number of Sentinel-2
  image pixels.

We can use all pixels in all available datasets, or we can just fit a portion of those pixels. Even if we don't use all
  pixels to fit the model, we can still later use the model to predict the cluster of any pixel.

We don't know the optimal number of clusters to fit, we will have to experiement with different values.
We train several models with different settings simulataneously to avoid repeating the time-consuming jp2 load process.

With 100 clusters and 1/10th batch size, a complete ESA set takes ~28 minutes to process. The entire 111 sets
 would take 55 hours!
  -If we just sample one batch per set, time per set drops to 165s, making total time ~5 hours.
  -The most significant bottlenecks are:
        *loading the jp2 files (~120s/batch)
        *fitting one 100 cluster MBKM model (~16s/batch)
        *loading and applying StandardScaler (~2s/batch)
        *Interpolating up to high rez (~22s/batch)
        *An additional ~3s/batch belongs to other operations.
  -Fitting time is longest in the first batch (~120s) drops significantly in subsequent batches(~16s)
  -Could try replacing the jp2 files with npy files for faster loading time (using dtype='uint16')
  -Could experiement with different batch sizes or cluster numbers to see if fit time improves.
  -Maybe the only viable option is to fit a sampling of our pixels, rather than trying to get every decent pixel in the entire set.
  -The need to do a lot of file loading probably makes this a complicated task for cloud computing.

Fitting time increases linearly with the number of clusters. (m=1.47s/cluster initial, m=0.0939s/cluster subsequent)
  -Initial fitting time can get very long, with the first set taking ~24mins for 1000 clusters.
  -Subsequent fits take just 12s for 60 clusters, 100s for 1000 clusters
  -See MBKM_times.xlsx for full details.

With 1/10th batch size, python uses ~1GB memory. Nearly all of this usage comes from band_img and spectral_pixels

!!These clusters dont seem to benefit the prediciton model very much. Try first balancing settlement examples before training clusters!!
"""
from satellite_head import *
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
from timeit import default_timer as timer
import pickle

#Hardcoded path to satellite data dir:
SAT_DATA_DIR = "./USAID_africa/ESA/"
#Hardcoded satellite image size:
SAT_SIZE = 10980
#Hardcoded spectral band names:
BAND_LIST = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12']
BAND_REZS = [60,   10,    10,    10,   20,   20,   20,   10,   20,   60,   60,   20,   20]
#Hardcoded number of clusters:
# N_CLUSTERS = 100
CLUSTER_NUMS = [60,120,250,500,1000]
#Hardcoded number of pixels per batch:
BSIZE = 12056040 #Exactly 1/10th of pixels in 10980x10980 image

###Go through each ESA dataset and partial_fit our MiniBatchKMeans:
#List of any folders in the SAT_DATA_DIR with the '.SAFE' extension:
sat_set_list = os.listdir(SAT_DATA_DIR)
sat_set_list = [s for s in sat_set_list if s[-5:]=='.SAFE']

###Initialize model:
# print("Creating {} cluster model with {} pixels per batch...".format(N_CLUSTERS,BSIZE))
# mbkm = cluster.MiniBatchKMeans(n_clusters=N_CLUSTERS, compute_labels=False)

#Use multiple models:
mbkm_list = []
for n in CLUSTER_NUMS:
    print("Creating {} cluster model with {} pixels per batch...".format(n,BSIZE))
    mbkm_list += [cluster.MiniBatchKMeans(n_clusters=n, compute_labels=False)]

###Fit data from each ESA dataset:
for i,folder in enumerate(sat_set_list):
    t1 = timer()
    print("Fitting dataset {} of {}: '{}'".format(i+1, len(sat_set_list), folder))
    #Find a box that avoids dead zones:
    [x0,x1,y0,y1] = box_in_footprint(SAT_DATA_DIR+folder)
    x0 += 100
    y0 += 100
    x1 -= 100
    y1 -= 100

    ###Just use one batch per dataset:
    batch_size = min([BSIZE, (y1-y0)*(x1-x0)])
    spectral_pixels = np.zeros((batch_size,len(BAND_LIST))).astype('float32')
    loading_jp2_time = 0
    ss_time = 0
    interpolation_time = 0
    for b,band in enumerate(BAND_LIST):
        print(band,end='|',flush=True)
        #Load up the band:
        t2 = timer()
        band_img = get_band(band,SAT_DATA_DIR+folder)
        loading_jp2_time += timer()-t2
        #If the image is not on the 10980x10980 grid, use bilinear interpolation to blow it up:
        if band_img.shape[0]!=SAT_SIZE:
            t2 = timer()
            band_img = scipy.misc.imresize(band_img,SAT_SIZE/band_img.shape[0],interp='bilinear',mode='F')
            interpolation_time += timer()-t2
        #Only keep the pixels belonging to this batch:
        band_img = band_img[y0:y1, x0:x1].reshape(-1,1)[0:batch_size]
        #Apply StandardScaler from the grand fit:
        t2 = timer()
        with open('./USAID_africa/StandardScalers/'+band+'_grand_SS.pkl','rb') as pfile:
            ss = pickle.load(pfile)
        band_img = ss.transform(band_img.astype('float32'))
        ss_time += timer()-t2
        #Put this band into our matrix:
        spectral_pixels[:,b] = band_img.reshape(-1)
    #Apply partial_fit:
    ##### Multiple MBKMs ##########
    print("  --JP2 loading: {:.1f}s, SS: {:.1f}s, Interpolation: {:.1f}s".format(loading_jp2_time, ss_time, interpolation_time))
    for a in range(len(mbkm_list)):
        t2 = timer()
        mbkm_list[a].partial_fit(spectral_pixels)
        print("{} cluster MBKM: {:.1f}s".format(CLUSTER_NUMS[a], timer()-t2))
        #Save after each dataset:
        filename = './USAID_africa/Models/MBKM_grand_{}.pkl'.format(CLUSTER_NUMS[a])
        with open(filename, 'wb') as pfile:
            pickle.dump(mbkm_list[a], pfile, pickle.HIGHEST_PROTOCOL)
    ############################
    ###Single MBKM:
    # t2 = timer()
    # mbkm.partial_fit(spectral_pixels)
    # print("  --JP2 loading: {:.1f}s, Partial fit: {:.1f}s,  SS: {:.1f}s, Interpolation: {:.1f}s".format(loading_jp2_time, timer()-t2, ss_time, interpolation_time))

    print("ESA set processed in {:.1f}s.".format((timer()-t1)/1))
    # #Save after each dataset:
    # filename = './USAID_africa/Models/MBKM_grand_{}.pkl'.format(N_CLUSTERS)
    # with open(filename, 'wb') as pfile:
    #     pickle.dump(mbkm, pfile, pickle.HIGHEST_PROTOCOL)

# ###Store fitted MBKM model to disk:
# print('Saving MBKM with pickle...')
# filename = './USAID_africa/Models/MBKM_grand_{}.pkl'.format(N_CLUSTERS)
# with open(filename, 'wb') as pfile:
#     pickle.dump(mbkm, pfile, pickle.HIGHEST_PROTOCOL)

#########################
myalert()
print('Done')



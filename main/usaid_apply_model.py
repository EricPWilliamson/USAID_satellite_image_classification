"""
Loads pickled classification and clustering models and then applies them to an entire ESA image. Labeling grids within
  the image as "Farmland", "Settlement", or "Wilderness".

The bottleneck of this script is predicting clusters with the Birch model. Memory usage is rather low at BSIZE=5000, so
  try going higher. Using an MBKM might be necessary for large scale deployment.
"""
from satellite_head import *
from timeit import default_timer as timer
from sklearn import cluster
from sklearn.ensemble import RandomForestClassifier
import pickle
plt.ion()

#Hardcoded path to satellite data set:
SAT_DATA_DIR = "./USAID_africa/ESA/S2A_MSIL1C_20170119T074231_N0204_R092_T37PDK_20170119T075325.SAFE"
#Hardcoded spectral band names:
BAND_LIST = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12']
BAND_REZS = [60,   10,    10,    10,   20,   20,   20,   10,   20,   60,   60,   20,   20]
#Hardcoded satellite image size:
SAT_SIZE = 10980
#Hardcoded area around each USAID point to be taken into consideration. Should be the same as was used to train the model.
VIEW_SIZE = 100

#Harcoded path to pickled StandardScalers dir:
SS_DIR = './USAID_africa/StandardScalers/'
#Path to pickled cluster model:
CLUSTER_PATH = './USAID_africa/premerged/birch_n250_rbt5_tr120pct.pkl'
#Batch size for cluster prediction:
BSIZE = 20000
#Path to pickled classification model:
CLASSIFIER_PATH = './USAID_africa/Models/classifier_RFC100_Birch250_20180830.pkl'
#Encoding for different class labels:
CLASS_LABELS = {'farmland':'F', 'settlement':'S', 'wilderness':'W'}
#Path for temporary files:
TMP_PATH = './usaid_apply_model_tmp/'
#Sometimes we can just reuse the temp files from the previous execution:
REUSE_TMP = True
#Location for output files:
OUTPUT_DIR = './USAID_africa/final_results/'

def check_corners(x0,x1,y0,y1):
    """Checks whether or not all 4 corners lie within the footprint found in SAT_DATA_DIR. Returns a boolean."""
    pxl_corners = [[x0,y0],
                   [x0,y1],
                   [x1,y0],
                   [x1,y1]]
    latlon_corners = np.zeros((4,2))
    for i,coord in enumerate(pxl_corners):
        latlon_corners[i,0],latlon_corners[i,1] = latlon_to_pixel(coord[0],coord[1],inverse=True,data_dir=SAT_DATA_DIR)
    return all(check_footprint(SAT_DATA_DIR,latlon_corners))

###Load the entire image and save each band image as a temporary file (not enough memory to load all bands at once):
if not REUSE_TMP:
    t1 = timer()
    for b,band in enumerate(BAND_LIST):
        print(band,end='|',flush=True)
        band_img = get_band(band,SAT_DATA_DIR)
        #If the image is not on the 10980x10980 grid, use bilinear interpolation to blow it up:
        if band_img.shape[0]!=SAT_SIZE:
            band_img = scipy.misc.imresize(band_img,SAT_SIZE/band_img.shape[0],interp='bilinear',mode='F')
        #Apply StandardScaler from the grand fit:
        with open(SS_DIR+band+'_grand_SS.pkl','rb') as pfile:
            ss = pickle.load(pfile)
        band_img = ss.transform(band_img.astype('float32'))
        #Save the band as a npy file for faster reloading:
        np.save(TMP_PATH+band+'.npy',band_img)
    del band_img #Deleting this var saves a lot of memory
    print("  --jp2 files loaded and scaled in {:.1f}s".format(timer()-t1),end='\n',flush=True)

###Load the pickled models:
with open(CLUSTER_PATH, 'rb') as pfile:
    cluster_model = pickle.load(pfile)
with open(CLASSIFIER_PATH, 'rb') as pfile:
    classifier_model = pickle.load(pfile)

###Process image one view area at at time:
grid_size = int(np.floor((SAT_SIZE-5) / (2*VIEW_SIZE))) #I use an extra 5 pixel buffer to keep the first row within the footprint.
# grid_size = 10 #Manually defined a smaller grid size for testing
class_grid = np.zeros((grid_size,grid_size),dtype=object)
for r in range(grid_size):
    for c in range(grid_size):
        #Define the corners of the view area:
        x0 = c * 2*VIEW_SIZE + 5
        x1 = x0 + 2*VIEW_SIZE
        y0 = r * 2*VIEW_SIZE + 5
        y1 = y0 + 2*VIEW_SIZE
        #Check that the view area is within the footprint:
        if check_corners(x0,x1,y0,y1):
            #Proceed to load the full spectral data:
            print('Loading view area {}/{}...'.format(r*grid_size+c,grid_size**2),end='',flush=True)
            t1 = timer()
            area_spectral = np.zeros((2*VIEW_SIZE, 2*VIEW_SIZE, len(BAND_LIST))).astype('float32')
            for b,band in enumerate(BAND_LIST):
                band_img = np.load(TMP_PATH+band+'.npy', mmap_mode='r')
                area_spectral[:,:,b] = band_img[y0:y1,x0:x1]
            del band_img
            print("took {:.1f}s".format(timer()-t1))

            #Predict cluster for each pixel:
            print('Applying cluster model...',end='',flush=True)
            t1 = timer()
            cluster_pred = np.zeros((2*VIEW_SIZE)**2,dtype='uint32')
            for p in range(0, (2*VIEW_SIZE)**2, BSIZE):
                cluster_pred[p:p+BSIZE] = cluster_model.predict(area_spectral.reshape(-1,13)[p:p+BSIZE,:])
            print('took {:.1f}s'.format(timer()-t1))
            #Save clusters to file:
            np.save(OUTPUT_DIR+'{}_{}_clusters.npy'.format(x0,y0), cluster_pred)
            if False:
                #Plot view area:
                print('Plotting clusters...')
                n_clusters = cluster_model.n_clusters
                figtitle = 'Sample clusters'
                np.random.seed(285)
                colors = np.random.rand(n_clusters,3).astype('float32')
                c = colors[ypred]
                imshow_to_file(c.reshape((2*VIEW_SIZE,2*VIEW_SIZE,3)),
                               'applied_clustering.png',
                               title=figtitle)

            #Tally up cluster populations within view area:
            print('Tallying cluster populations...',end='',flush=True)
            cluster_pops = np.zeros(cluster_model.n_clusters,dtype='uint32')
            unique,counts = np.unique(cluster_pred,return_counts=True)
            cluster_pops[unique] = counts
            print('Predominant cluster: #{}'.format(np.argmax(cluster_pops)))

            #Predict classification and convert it to an integer:
            print('Predicting class...',end='',flush=True)
            class_pred = classifier_model.predict(cluster_pops.reshape(1,-1))[0]
            print(class_pred)
            class_grid[r,c] = CLASS_LABELS[class_pred]
np.save(OUTPUT_DIR+'class_grid.npy', class_grid)

###Plot the results over the TCI image:
print('Creating RGB image...')
tci = get_band("TCI",SAT_DATA_DIR)
plt.figure()
plt.title('Classification Results')
plt.imshow(tci)
for j in range(grid_size+1):
    plt.axhline(y=j*2*VIEW_SIZE+5,color='gray')
    plt.axvline(x=j*2*VIEW_SIZE+5,color='gray')
for r in range(grid_size):
    for c in range(grid_size):
        plt.text(c*2*VIEW_SIZE+5, r*2*VIEW_SIZE+5, str(class_grid[r,c]), color='w', verticalalignment='top')

print('Done')

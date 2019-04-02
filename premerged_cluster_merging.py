"""
Takes in premerged data, uses pickled clustering models to predict clusters, tallies up cluster populations, exports a
  dataframe to csv that can be fed into our classification models.

Memory usage is much higher for birch models with large number of subclusters, requiring us to use smaller batches.
  Reducing batch size from 1M to 10k achieved this and didnt have a significant impact on prediction time with the
  other clustering models.

The high threshold birches are fast: prediction time 1~5 minutes (more clusters take longer)
The low threshold birches are slow: prediciton time can take 3.5 hours for one model
The MKBKs are fast: prediction time 2~10 minutes (they can take larger batches and perhaps go faster)
"""
import pandas as pd
from satellite_head import *
from timeit import default_timer as timer
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
import pickle

#Hardcoded spectral band names:
BAND_LIST = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12']
BAND_REZS = [60,   10,    10,    10,   20,   20,   20,   10,   20,   60,   60,   20,   20]
#Harcoded path to pickled StandardScalers dir:
SS_DIR = './USAID_africa/StandardScalers/'
#Hardcoded file path for dataframe:
USAID_DATA_PATH = './USAID_africa/premerge_data_rbt5.csv'
#Hardcoded file path for big npy pixel matrix:
PXL_NPY_PATH = './USAID_africa/premerge_data_rbt5.npy'
#Hardcoded dir for output files:
OUT_DIR = './USAID_africa/premerged/merged_csvs/'
#Hardcoded dir for pickled, prefit clustering models:
CMODEL_DIR = './USAID_africa/Models/test_roster2/'
#Batch size for predicting clusters:
BSIZE = 10000
#Number of pixels in each view area:
VIEWSIZE = 200*200

t0 = timer()
###Load datafiles:
master_df = pd.read_csv(USAID_DATA_PATH)
print('Loading pxl_mat from .npy file...')
t1 = timer()
pxl_mat = np.load(PXL_NPY_PATH)
print('Loading took {:.1f}s'.format(timer()-t1))
#Reshape:
pxl_mat = pxl_mat.reshape(-1,13)
#Verify amount of data:
if pxl_mat.shape[0] != master_df.shape[0]*VIEWSIZE:
    raise Exception('Number of pixels loaded from npy file does not match the number of view areas listed in csv file.')
#Apply StandardScaler:
print('Applying SS to {:.1f} M pixels ({:.1f}% of a full ESA image)...'.format(pxl_mat.shape[0]/10**6, pxl_mat.shape[0]*100/10980**2))
t1 = timer()
for b,band in enumerate(BAND_LIST):
    with open(SS_DIR+band+'_grand_SS.pkl','rb') as pfile:
        ss = pickle.load(pfile)
    pxl_mat[:,b] = ss.transform(pxl_mat[:,b].reshape(-1, 1)).reshape(-1)
print('StandardScaler took {:.1f}s.'.format(timer()-t1))

###Go through each clustering model:
cmodel_files = os.listdir(CMODEL_DIR)
for cmodel_file in cmodel_files:
    outpath = '{}merged_rbt5_with_{}.csv'.format(OUT_DIR,cmodel_file[:-4])
    if os.path.exists(outpath):
        print('\n' + outpath + ' already exists.')
    else:
        print('\n===Using ' + cmodel_file)
        try:
            #Load the pickle:
            with open(CMODEL_DIR+cmodel_file,'rb') as pfile:
                cmodel = pickle.load(pfile)
            print(cmodel)
            n_clusters = cmodel.n_clusters

            #Predict cluster of each pixel, using batches to limit memory usage:
            print('Predicting cluster labels...', end='', flush=True)
            t1 = timer()
            label_list = np.zeros(pxl_mat.shape[0], dtype='uint32')
            for p in range(0,pxl_mat.shape[0],BSIZE):
                print('\rPredicting cluster labels... {:.2f}%'.format(100*p/pxl_mat.shape[0]), end='', flush=True)
                label_list[p:p+BSIZE] = cmodel.predict(pxl_mat[p:p+BSIZE])
            print('\rCluster prediction took {:.1f}s.'.format(timer()-t1))

            #Look at each view area and tally cluster populations:
            print('Tallying cluster populations...')
            cluster_pops = np.zeros((int(pxl_mat.shape[0]/VIEWSIZE), n_clusters), dtype='uint32')
            for a,p in enumerate(range(0,pxl_mat.shape[0],VIEWSIZE)):
                unique,counts = np.unique(label_list[p:p+VIEWSIZE],return_counts=True)
                cluster_pops[a,unique] = counts

            #Assemble DataFrame and save to csv:
            print('Saving csv...')
            old_cols = list(master_df.columns.values)
            cluster_cols = ['C_{}'.format(n+1) for n in range(n_clusters)]
            merged_df = pd.DataFrame(np.append(master_df.values,cluster_pops,axis=1), columns=old_cols+cluster_cols)
            merged_df.to_csv(outpath, index=False)
        except: # catch *all* exceptions
            print('')
            print(sys.exc_info()[0])
            print('Skipping to next model...')

print('Total time: {:.2f} hours.'.format((timer()-t0)/3600))
myalert()
print('Done')

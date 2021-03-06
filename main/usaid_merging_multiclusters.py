"""
Tweaked version of usaid_merging.py. Set up for creating multiple output files, each one created from a different
  pickled clustering algorithm.

Prediction times increase with n_clusters, but not linearly. 1000 clusters only takes ~3 times as long as 60 clusters.

Running all 111 datasets through clusters 60,250,500,1000 took ~10 hours.
"""
import pandas as pd
from satellite_head import *
from timeit import default_timer as timer
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
import pickle

#Hardcoded path to satellite data dir:
SAT_DATA_DIR = "./USAID_africa/ESA/"
#Hardcoded file path for input file:
USAID_DATA_PATH = './USAID_africa/tidy_data.csv'
#Hardcoded satellite image size:
SAT_SIZE = 10980
#Hardcoded list of spectral bands:
BAND_LIST = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12']

#Hardcoded area around each USAID point to be taken into consideration. We will assume it's a square, but it is not clear how large an area the user considered.
VIEW_SIZE = 100 #!!I will just assume its about 200x200 pixels. This can be altered later as part of the optimization process.
#Hardcoded paths to pickled models:
MBKM_PATHS = ['./USAID_africa/Models/MBKM_grand_60.pkl',
              './USAID_africa/Models/MBKM_grand_250.pkl',
              './USAID_africa/Models/MBKM_grand_500.pkl',
              './USAID_africa/Models/MBKM_grand_1000.pkl']
#Harcoded path to pickled StandardScalers dir:
SS_DIR = './USAID_africa/StandardScalers/'
#Flag for cluster method
USE_CLUSTER = True
#Hardcoded file path for output file:
OUT_PATH = './USAID_africa/cluster_merged_data_n'

def cluster_populations_multi(sat_set_path, match_df, cluster_algs):
    """
    A re-write of cluster_populations() that uses multiple models and returns a list of dataframes (one for each model).
    """
    ###Get the USAID point coordinates:
    pxl_points = match_df[['pxlX','pxlY']].values.tolist()
    ###We need to load jp2 files and use the MBKM to predict labels:
    #Load all the relevant pixels within this dataset:
    full_pixels = np.zeros((len(pxl_points),(2*VIEW_SIZE)**2,len(BAND_LIST)), dtype=np.float32)
    for b,band in enumerate(BAND_LIST):
        print(band,end='|',flush=True)
        #Load up the band and corresponding grand SS:
        with open(SS_DIR+band+'_grand_SS.pkl','rb') as pfile:
            ss = pickle.load(pfile)
        band_img = get_band(band,sat_set_path)
        #If the image is not on the 10980x10980 grid, use bilinear interpolation to blow it up:
        if band_img.shape[0]!=SAT_SIZE:
            band_img = scipy.misc.imresize(band_img,SAT_SIZE/band_img.shape[0],interp='bilinear',mode='F').astype('float32')
        #Crop the image around each USAID point, apply SS, and aggregate the data:
        for p,pair in enumerate(pxl_points):
            x0 = int(np.round(pair[0]-VIEW_SIZE))
            x1 = int(np.round(pair[0]+VIEW_SIZE))
            y0 = int(np.round(pair[1]-VIEW_SIZE))
            y1 = int(np.round(pair[1]+VIEW_SIZE))
            img_crop = band_img[y0:y1,x0:x1]
            img_crop = ss.transform(img_crop.astype('float32'))
            full_pixels[p,:,b] = img_crop.reshape(-1)
    print(' -OK')
    ###Put the full pixels through each of our cluster algorithms:
    old_cols = list(match_df.columns.values)
    df_mat = match_df.values
    match_df_list = []
    for cluster_alg in cluster_algs:
        t3 = timer()
        #Get n_clusters:
        n_clusters = cluster_alg.n_clusters
        cluster_cols = ['C_{}'.format(n+1) for n in range(n_clusters)]
        #Predict labels in each crop and convert to cluster population:
        cluster_pops = np.zeros((len(pxl_points),n_clusters),dtype=np.int)
        for p in range(len(pxl_points)):
            label_list = cluster_alg.predict(full_pixels[p,:,:])
            unique,counts = np.unique(label_list,return_counts=True)
            cluster_pops[p,unique] = counts
        #Create a new df that includes cluster pop features:
        big_mat = np.append(df_mat,cluster_pops,axis=1)
        #Append that df to a list of dfs:
        match_df_list += [pd.DataFrame(big_mat,columns=old_cols+cluster_cols)]
        print("Predicted {} clusters in {:.2f}s".format(n_clusters,timer()-t3))
    return match_df_list

def cluster_populations(sat_set_path, match_df, cluster_alg):
    """
    An advanced approach that finds the cluster populations within the view area according to a pickled cluster model.
    Looks at a square area around each USAID point and fits the pixels with a pickled cluster model. The new features
    it generates are the population counts of the different clusters. This should be much more effective than the
    "blender" method.
    """
    ###Get the USAID point coordinates:
    pxl_points = match_df[['pxlX','pxlY']].values.tolist()
    ###Get n_clusters:
    n_clusters = cluster_alg.n_clusters
    cluster_cols = ['C_{}'.format(n+1) for n in range(n_clusters)]
    ###Get the full spectral pixels in each view area:
    #Check to see if pixel labels have already been predicted:
    sat_set_name = sat_set_path.split('/')[-1]
    if len(sat_set_name)==0:
        sat_set_name = sat_set_path.split('/')[-2]
    try:
        npy_path = find_file(sat_set_name+'*.npy', LABEL_NPY_DIR)
        print('Found pre-computed pixel labels.')
    except Exception:
        print('No saved labels were found. Proceed to load sat image and predict clusters...')
        npy_path = None

    if npy_path is not None:
        ###Just use pre-computed pixel labels to get cluster pop:
        pixel_labels = np.load(npy_path)
        cluster_pops = np.zeros((len(pxl_points),n_clusters), dtype=np.int)
        for p,pair in enumerate(pxl_points):
            x0 = int(np.round(pair[0]-VIEW_SIZE))
            x1 = int(np.round(pair[0]+VIEW_SIZE))
            y0 = int(np.round(pair[1]-VIEW_SIZE))
            y1 = int(np.round(pair[1]+VIEW_SIZE))
            label_list = pixel_labels[y0:y1,x0:x1].reshape(-1)
            unique,counts = np.unique(label_list,return_counts=True)
            cluster_pops[p,unique] = counts
    else:
        ###We need to load jp2 files and use the MBKM to predict labels:
        #Load all the relevant pixels within this dataset:
        full_pixels = np.zeros((len(pxl_points),(2*VIEW_SIZE)**2,len(BAND_LIST)), dtype=np.float32)
        for b,band in enumerate(BAND_LIST):
            print(band,end='|',flush=True)
            #Load up the band and corresponding grand SS:
            with open(SS_DIR+band+'_grand_SS.pkl','rb') as pfile:
                ss = pickle.load(pfile)
            band_img = get_band(band,sat_set_path)
            #If the image is not on the 10980x10980 grid, use bilinear interpolation to blow it up:
            if band_img.shape[0]!=SAT_SIZE:
                band_img = scipy.misc.imresize(band_img,SAT_SIZE/band_img.shape[0],interp='bilinear',mode='F').astype('float32')
            #Crop the image around each USAID point, apply SS, and aggregate the data:
            for p,pair in enumerate(pxl_points):
                x0 = int(np.round(pair[0]-VIEW_SIZE))
                x1 = int(np.round(pair[0]+VIEW_SIZE))
                y0 = int(np.round(pair[1]-VIEW_SIZE))
                y1 = int(np.round(pair[1]+VIEW_SIZE))
                img_crop = band_img[y0:y1,x0:x1]
                img_crop = ss.transform(img_crop.astype('float32'))
                full_pixels[p,:,b] = img_crop.reshape(-1)
        print(' -OK')
        #Predict labels in each crop and convert to cluster population:
        cluster_pops = np.zeros((len(pxl_points),n_clusters), dtype=np.int)
        for p in range(len(pxl_points)):
            label_list = cluster_alg.predict(full_pixels[p,:,:])
            unique,counts = np.unique(label_list,return_counts=True)
            cluster_pops[p,unique] = counts

    #Replace match_df with a df that includes cluster pop features
    old_cols = list(match_df.columns.values)
    df_mat = match_df.values
    big_mat = np.append(df_mat,cluster_pops,axis=1)
    match_df = pd.DataFrame(big_mat,columns=old_cols+cluster_cols)
    return match_df

def pixel_blender(sat_set_path, match_df):
    """A simple approach that just takes the average and std of the band pixel values surrounding each USAID point."""
    #Get the USAID point coordinates:
    pxl_points = match_df[['pxlX','pxlY']].values.tolist()
    #Get the mean and std one band at a time:
    for band in BAND_LIST:
        print(band, end='|', flush=True)
        #Load up the band:
        band_img = get_band(band,sat_set_path)
        #If the image is not on the 10980x10980 grid, use bilinear interpolation to blow it up:
        if band_img.shape[0]!=SAT_SIZE:
            band_img = scipy.misc.imresize(band_img,SAT_SIZE/band_img.shape[0],interp='bilinear',mode='F')
        #Crop the image around each USAID point and get distilled values for each:
        band_mean = []
        band_std = []
        for pair in pxl_points:
            x0 = int(np.round(pair[0]-VIEW_SIZE))
            x1 = int(np.round(pair[0]+VIEW_SIZE))
            y0 = int(np.round(pair[1]-VIEW_SIZE))
            y1 = int(np.round(pair[1]+VIEW_SIZE))
            img_crop = band_img[y0:y1,x0:x1]
            #Some points will be too close to the borders of the satellite image, so we can't get the whole crop:
            if img_crop.shape!=(2*VIEW_SIZE,2*VIEW_SIZE):
                print('ERROR:')
                print(img_crop.shape)
            band_mean += [np.mean(img_crop)]
            band_std += [np.std(img_crop)]
            #Check for strange problem with zeros:
            if np.mean(img_crop) < .1:
                print('Dark image problem!')
        #Put new features into dataframe:
        match_df[band+'_mean'] = band_mean
        match_df[band+'_std'] = band_std
    print(' -OK')
    return match_df

def get_image_crops(usaid_df,sat_set_path, cluster_alg):
    #Only keep the points that match the Sentinel-2 image area:
    coords = usaid_df[['X','Y']].values.tolist()
    is_in_footprint = check_footprint(sat_set_path,coords)
    match_df = usaid_df[is_in_footprint]
    print("{} of {} points from USAID set are included in this footprint.".format(match_df.shape[0],usaid_df.shape[0]))
    coords = match_df[['X','Y']].values.tolist()

    ###Look up locations in match_df and add the pixel values as new columns:
    pxl_points = []
    for pair in coords:
        pxl_x,pxl_y = latlon_to_pixel(pair[0],pair[1],inverse=False,data_dir=sat_set_path)
        pxl_points += [[pxl_x,pxl_y]]
    pxl_points = np.array(pxl_points)
    match_df = match_df.assign(pxlX=pxl_points[:,0])
    match_df = match_df.assign(pxlY=pxl_points[:,1])

    #Remove any points that are so close to the edge of our satellite image that we can't get the entire view:
    match_df = match_df[match_df['pxlX']>VIEW_SIZE]
    match_df = match_df[match_df['pxlX']<(SAT_SIZE-VIEW_SIZE)]
    match_df = match_df[match_df['pxlY']>VIEW_SIZE]
    match_df = match_df[match_df['pxlY']<(SAT_SIZE-VIEW_SIZE)]
    print("{} points remain after applying border buffer.".format(match_df.shape[0]))

    print("Processing satellite image...")
    if USE_CLUSTER:
        if type(cluster_alg) is list:
            return cluster_populations_multi(sat_set_path, match_df, cluster_alg)
        else:
            match_df = cluster_populations(sat_set_path, match_df, cluster_alg)
    else:
        match_df = pixel_blender(sat_set_path, match_df)
    return match_df


###Load tidy USAID data:
master_df = pd.read_csv(USAID_DATA_PATH)
###Load the MBKMs from pickles:
mbkms = []
merged_df = []
n_list = []
for mpath in MBKM_PATHS:
    with open(mpath,'rb') as pfile:
        mbkm = pickle.load(pfile)
    mbkms += [mbkm]
    merged_df += [None]
    n_list += [mbkm.n_clusters]

###Go through each ESA dataset and create merged USAID-ESA data for each:
#List of any folders in the SAT_DATA_DIR with the '.SAFE' extension:
sat_set_list = os.listdir(SAT_DATA_DIR)
sat_set_list = [s for s in sat_set_list if s[-5:]=='.SAFE']
for i,folder in enumerate(sat_set_list):
    t1 = timer()
    print("\nMerging dataset {} of {}: '{}'".format(i+1, len(sat_set_list), folder))
    #Get the dataframe for any USAID points contained in this dataset area:
    ###Multiple mbkms:
    new_df_list = get_image_crops(master_df,SAT_DATA_DIR+folder,mbkms)
    for j,new_df in enumerate(new_df_list):
        #Add new data to composite dataframe:
        if merged_df[j] is None:
            merged_df[j] = new_df.copy()
        else:
            merged_df[j] = merged_df[j].append(new_df)
        #Save each time:
        merged_df[j].to_csv(OUT_PATH+str(n_list[j])+'.csv',index=False)
    print("ESA set processed in {:.2f} minutes.".format((timer()-t1)/60))

myalert()
print('Done')

"""
Load a subset of of the USAID points, such that the farmland/settlement/wilderness classes are balanced. Then, since the
  subset is small enough, load all the pixels from the view area around those points and save everything in a npy file.
  This npy file can be quickly loaded later on for distilling merged features (mainly saves us the trouble of loading so
  many jp2 files.)

Took 6 hours for 2016 points. File size came out to be ~4GB. Memory maxed out towards end of set list, causing slow downs.
4GB npy files are a little too big to work with. 2GB is much much faster.

!!Inefficient to put everything into one file. Try assembling several manageable files that can be combined later.
!!Saving in every iteration uses a significant amount of time, just save every 10th iteration or something.
!!Balancing before we know which points will be eliminated by border conditions is inaccurate. We can first load a
    dataframe that's already had edge points removed, so that we know the make-up of classes wont be altered.
"""
import pandas as pd
from satellite_head import *
from timeit import default_timer as timer

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
#Hardcoded balance tolerance factor:
BALANCE_TOL = 10
#Hardcoded file path for output file:
OUT_PATH = './USAID_africa/premerge_data_t{}'.format(BALANCE_TOL)

def balance_classes(df, btol):
    """Even up the support of the classes."""
    #Find the least supported class and muliply by the tolerance coefficient to get max_count:
    ccounts = df['classification'].value_counts()
    max_count = np.min(ccounts.values) * btol
    #Create a new dataframe with balenced support:
    newdf = pd.DataFrame(columns=df.columns.values)
    for x in df.groupby('classification'):
        if x[1].shape[0] > max_count:
            newdf = newdf.append(x[1].sample(max_count).reset_index(drop=True))
        else:
            newdf = newdf.append(x[1].reset_index(drop=True))
    return newdf.reset_index(drop=True)

def get_image_crops(usaid_df, sat_set_path):
    #Only keep the points that match the Sentinel-2 image area:
    coords = usaid_df[['X','Y']].values.tolist()
    is_in_footprint = check_footprint(sat_set_path,coords)
    match_df = usaid_df[is_in_footprint]
    print("{} of {} points from USAID subset are included in this footprint.".format(match_df.shape[0],usaid_df.shape[0]))
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
    pxl_points = match_df[['pxlX','pxlY']].values.tolist()

    print("Processing satellite image...")
    ###We need to load jp2 files and use the MBKM to predict labels:
    #Load all the relevant pixels within this dataset:
    full_pixels = np.zeros((len(pxl_points),(2*VIEW_SIZE)**2,len(BAND_LIST)),dtype=np.float32)
    for b,band in enumerate(BAND_LIST):
        print(band,end='|',flush=True)
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
            full_pixels[p,:,b] = img_crop.reshape(-1)
    print(' -OK')
    return match_df,full_pixels


t0 = timer()
###Load tidy USAID data and take a balanced subset:
master_df = pd.read_csv(USAID_DATA_PATH)
#Rebalance the data:
master_df = balance_classes(master_df,BALANCE_TOL)
print("Selected {} USAID points.".format(master_df.shape[0]))
data_est = master_df.shape[0] * (2*VIEW_SIZE)**2 * len(BAND_LIST)
print('Expected memory usage: {:.1f}MB'.format(data_est*4/2**20))

###Go through each ESA dataset and create merged USAID-ESA data for each:
merged_df = None
pxl_mat = np.array([])
#List of any folders in the SAT_DATA_DIR with the '.SAFE' extension:
sat_set_list = os.listdir(SAT_DATA_DIR)
sat_set_list = [s for s in sat_set_list if s[-5:]=='.SAFE']
for i,folder in enumerate(sat_set_list):
    t1 = timer()
    print("\nMerging dataset {} of {}: '{}'".format(i+1, len(sat_set_list), folder))
    #Get the view areas for any USAID points contained in this dataset area:
    new_df,new_pixels = get_image_crops(master_df,SAT_DATA_DIR+folder)
    #Add new data to composite dataframe:
    if merged_df is None:
        merged_df = new_df.copy()
        pxl_mat = np.copy(new_pixels)
    else:
        merged_df = merged_df.append(new_df)
        pxl_mat = np.append(pxl_mat,new_pixels,axis=0)
    #Save each time:
    merged_df.to_csv(OUT_PATH+'.csv',index=False)
    np.save(OUT_PATH+'.npy', pxl_mat)
    print("ESA set processed in {:.2f} minutes.".format((timer()-t1)/60))
print('Total time: {:.2f} hours'.format((timer()-t0)/3600))

myalert()
print('DONE')



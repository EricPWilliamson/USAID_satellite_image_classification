"""
Takes up where usaid_wrangling.py left off. Merges usaid data with corresponding data from Sentinel-2.

Rather than preparing data for random forest or the like, we will prepare data for a Keras DNN model.

The simplest RGB method only loads the actual RGB color channels.
!!Introduce a more sophisticated method later.

!!Note that ESA images are on (y,x) coordinates
!!Optimize model on VIEW_SIZE since we dont know how much area the users considered when making their classifiction
"""
import pandas as pd
from satellite_head import *
from timeit import default_timer as timer
from imageio import imwrite


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
#Flag for colorization method
COLOR_METHOD = 'simple RGB'
#Hardcoded file path for output file:
OUT_DIR = './USAID_africa/simple_RGB/'

def store_rgb(sat_set_path, match_df):
    """A simple approach that just takes the red (B04), green (B03), and blue (B02) bands to create an image file."""
    #Get the USAID point coordinates:
    pxl_points = match_df[['pxlX','pxlY']].values.tolist()
    classifications = match_df['classification'].values.tolist()
    image_mat = np.zeros((2*VIEW_SIZE, 2*VIEW_SIZE, 3, len(pxl_points)))
    #Get the mean and std one band at a time:
    for b,band in enumerate(['B04','B03','B02']):
        print(band, end='|', flush=True)
        #Load up the band:
        band_img = get_band(band,sat_set_path)
        #If the image is not on the 10980x10980 grid, use bilinear interpolation to blow it up:
        if band_img.shape[0]!=SAT_SIZE:
            band_img = scipy.misc.imresize(band_img,SAT_SIZE/band_img.shape[0],interp='bilinear',mode='F')
        #Crop the image around each USAID point and save an image for each:
        for p,pair in enumerate(pxl_points):
            x0 = int(np.round(pair[0]-VIEW_SIZE))
            x1 = int(np.round(pair[0]+VIEW_SIZE))
            y0 = int(np.round(pair[1]-VIEW_SIZE))
            y1 = int(np.round(pair[1]+VIEW_SIZE))
            img_crop = band_img[y0:y1,x0:x1]
            #Some points will be too close to the borders of the satellite image, so we can't get the whole crop:
            if img_crop.shape!=(2*VIEW_SIZE,2*VIEW_SIZE):
                raise Exception('Error: Image crop shape is ' + str(img_crop.shape))
            #append band to 3 channel matrix:
            image_mat[:,:,b,p] = img_crop
    #Save images:
    filename_list = []
    for c,classification in enumerate(classifications):
        idx = len([name for name in os.listdir(OUT_DIR+classification) if '.png' in name])
        out_path = "{0:s}{1:s}/{1:s}.{2:d}.png".format(OUT_DIR,classification,idx)
        filename_list += ["{:s}.{:d}.png".format(classification,idx)]
        imwrite(out_path, scale_down(image_mat[:,:,:,c], scale_factor=32), format='png')
    match_df['filename'] = filename_list
    print(' -OK')
    return match_df


def get_image_crops(usaid_df, sat_set_path):
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
    if COLOR_METHOD == 'simple RGB':
        match_df = store_rgb(sat_set_path, match_df)
    else:
        pass
        #!!!
    return match_df

###Load tidy USAID data:
master_df = pd.read_csv(USAID_DATA_PATH)

###Make sure output directories are ready:
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
for s in master_df['classification'].unique():
    if not os.path.exists(OUT_DIR+s):
        os.makedirs(OUT_DIR+s)

###Go through each ESA dataset and create merged USAID-ESA data for each:
merged_df = None
#List of any folders in the SAT_DATA_DIR with the '.SAFE' extension:
sat_set_list = os.listdir(SAT_DATA_DIR)
sat_set_list = [s for s in sat_set_list if s[-5:]=='.SAFE']
# sat_set_list = sat_set_list[108:]
for i,folder in enumerate(sat_set_list):
    t1 = timer()
    print("\nMerging dataset {} of {}: '{}'".format(i+1, len(sat_set_list), folder))
    #Get the dataframe for any USAID points contained in this dataset area:
    new_df = get_image_crops(master_df, SAT_DATA_DIR+folder)
    #Add new data to composite dataframe:
    if merged_df is None:
        merged_df = new_df.copy()
    else:
        merged_df = merged_df.append(new_df)
    print("ESA set processed in {:.2f} minutes.".format((timer()-t1)/60))
    #Save each time:
    merged_df.to_csv(OUT_DIR+'summary.csv',index=False)

###Save the merged data:
merged_df.to_csv(OUT_DIR+'summary.csv',index=False)

myalert()
print('Done')


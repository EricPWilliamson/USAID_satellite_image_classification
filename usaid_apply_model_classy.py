"""
Use a class object as a more convenient test bed for ML, pre-processing, etc.
"""
from satellite_head import *
from timeit import default_timer as timer
from sklearn import cluster
from sklearn.ensemble import RandomForestClassifier
import pickle
import pandas as pd
import matplotlib.patheffects as PathEffects
import matplotlib.colors as pltcolors
plt.ion()

#Hardcoded path to satellite data set:
SAT_DATA_DIR = "./USAID_africa/ESA/S2A_MSIL1C_20170119T074231_N0204_R092_T37PDK_20170119T075325.SAFE"
#Hardcoded spectral band names:
BAND_LIST = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12']
BAND_REZS = [60,10,10,10,20,20,20,10,20,60,60,20,20]
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
CLASS_LABELS = {'farmland':'F','settlement':'S','wilderness':'W'}
#Path for temporary files:
TMP_PATH = './usaid_apply_model_tmp/'
#Sometimes we can just reuse the temp files from the previous execution:
REUSE_TMP = True
#Location for output files:
OUTPUT_DIR = './USAID_africa/final_results/'

class ApplyModel(object):
    def __init__(self):
        self.class_grid = None
        self.tci = None
        self.cluster_map = None
        self.cluster_colors = None
        #Load the pickled models:
        with open(CLUSTER_PATH,'rb') as pfile:
            self.cluster_model = pickle.load(pfile)
        with open(CLASSIFIER_PATH,'rb') as pfile:
            self.classifier_model = pickle.load(pfile)

    @staticmethod
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

    @staticmethod
    def generate_tmp_files():
        """
        Loads the entire image and saves each band image as a temporary npy file that can be quickly recalled. There is
          not enough memory to load all bands at once, so we load one at a time.
        """
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
        del band_img #Deleting this var saves a lot of memory (maybe unnecessary here)
        print("  --jp2 files loaded and scaled in {:.1f}s".format(timer()-t1),end='\n',flush=True)

    def predict_class_grid(self):
        """Processes the image one view area at at time, predicts classification of that view area"""
        grid_size = int(np.floor((SAT_SIZE-5)/(2*VIEW_SIZE))) #I use an extra 5 pixel buffer to keep the first row within the footprint.
        # grid_size = 10 #Manually defined a smaller grid size for testing
        self.class_grid = np.zeros((grid_size,grid_size),dtype=object)
        for r in range(grid_size):
            for c in range(grid_size):
                #Define the corners of the view area:
                x0 = c*2*VIEW_SIZE+5
                x1 = x0+2*VIEW_SIZE
                y0 = r*2*VIEW_SIZE+5
                y1 = y0+2*VIEW_SIZE
                #Check that the view area is within the footprint:
                if self.check_corners(x0,x1,y0,y1):
                    #Proceed to load the full spectral data:
                    print('Loading view area {}/{}...'.format(r*grid_size+c,grid_size**2),end='',flush=True)
                    t1 = timer()
                    area_spectral = np.zeros((2*VIEW_SIZE,2*VIEW_SIZE,len(BAND_LIST))).astype('float32')
                    for b,band in enumerate(BAND_LIST):
                        band_img = np.load(TMP_PATH+band+'.npy',mmap_mode='r')
                        area_spectral[:,:,b] = band_img[y0:y1,x0:x1]
                    del band_img
                    print("took {:.1f}s".format(timer()-t1))

                    #Predict cluster for each pixel:
                    print('Applying cluster model...',end='',flush=True)
                    t1 = timer()
                    cluster_pred = np.zeros((2*VIEW_SIZE)**2,dtype='uint16')
                    for p in range(0,(2*VIEW_SIZE)**2,BSIZE):
                        cluster_pred[p:p+BSIZE] = self.cluster_model.predict(area_spectral.reshape(-1,13)[p:p+BSIZE,:])
                    print('took {:.1f}s'.format(timer()-t1))
                    #Save clusters to file:
                    np.save(OUTPUT_DIR+'{}_{}_clusters.npy'.format(x0,y0),cluster_pred)
                    if False:
                        #Plot view area:
                        print('Plotting clusters...')
                        n_clusters = self.cluster_model.n_clusters
                        figtitle = 'Sample clusters'
                        np.random.seed(285)
                        colors = np.random.rand(n_clusters,3).astype('float32')
                        c = colors[cluster_pred]
                        imshow_to_file(c.reshape((2*VIEW_SIZE,2*VIEW_SIZE,3)),'applied_clustering.png',title=figtitle)

                    #Tally up cluster populations within view area:
                    print('Tallying cluster populations...',end='',flush=True)
                    cluster_pops = np.zeros(self.cluster_model.n_clusters,dtype='uint32')
                    unique,counts = np.unique(cluster_pred,return_counts=True)
                    cluster_pops[unique] = counts
                    print('Predominant cluster: #{}'.format(np.argmax(cluster_pops)))

                    #Predict classification and convert it to an integer:
                    print('Predicting class...',end='',flush=True)
                    class_pred = self.classifier_model.predict(cluster_pops.reshape(1,-1))[0]
                    print(class_pred)
                    self.class_grid[r,c] = CLASS_LABELS[class_pred]
        np.save(OUTPUT_DIR+'class_grid.npy',self.class_grid)

    def load_class_grid(self,filepath):
        """Loads a stored class_grid from a npy file."""
        self.class_grid = np.load(filepath)

    def plot_class_grid(self):
        ###Plot the results over the TCI image:
        print('Creating RGB image...')
        if self.tci is None:
            self.tci = get_band("TCI",SAT_DATA_DIR)
        plt.figure()
        plt.title('Classification Results')
        plt.imshow(self.tci)
        self.overlay_class_grid()

    def overlay_class_grid(self):
        """
        Plots all view areas with grid lines and with color-coded labels showing the predicted classes. Can be called
          after plotting a map to produce an overlay effect.
        """
        grid_size = self.class_grid.shape[0]
        for j in range(grid_size+1):
            plt.axhline(y=j*2*VIEW_SIZE+5,color='gray')
            plt.axvline(x=j*2*VIEW_SIZE+5,color='gray')
        for r in range(grid_size):
            for c in range(grid_size):
                s = str(self.class_grid[r,c])
                if s=='F':
                    color = 'yellow'
                elif s=='S':
                    color = 'blue'
                elif s=='W':
                    color = 'white'
                else:
                    color = 'gray'
                txt = plt.text(c*2*VIEW_SIZE+5,r*2*VIEW_SIZE+5,
                               s,
                               color=color,
                               verticalalignment='top',
                               clip_on=True)
                txt.set_path_effects([PathEffects.withStroke(linewidth=1,foreground='k')])

    def compose_cluster_map(self,filedir):
        """Loads predicted clusters from multiple npy files. Assembles the view areas into a large composite map."""
        self.cluster_map = np.zeros((SAT_SIZE,SAT_SIZE),dtype='uint16')
        #Search the directory for all matching npy files:
        file_list = os.listdir(filedir)
        for file in file_list:
            file_parts = file.split('_')
            if len(file_parts)==3 and file_parts[2]=='clusters.npy':
                x0 = int(file_parts[0])
                y0 = int(file_parts[1])
                x1 = x0 + 2*VIEW_SIZE
                y1 = y0 + 2*VIEW_SIZE
                cluster_area = np.load(filedir+file).reshape(2*VIEW_SIZE,2*VIEW_SIZE)
                self.cluster_map[y0:y1,x0:x1] = cluster_area
        np.save(filedir+'cluster_map.npy',self.cluster_map)

    def load_cluster_map(self,filepath):
        """Loads a map that was already composed."""
        self.cluster_map = np.load(filepath)

    def plot_cluster_map(self, color_scheme='random'):
        """
        Plots the cluster map of the whole image. The boolean flag use_random_colors can help provide better visual
          contrast between different clusters.
        """
        plt.figure()
        plt.title('Cluster Map')
        if color_scheme=='random':
            #Randomly generate a color for each cluster. Usually provides more visual contrast between clusters.
            # np.random.seed(68454)
            colors = np.random.rand(self.cluster_model.n_clusters,3).astype('float32')
            color_map = colors[self.cluster_map.reshape(-1)].reshape(*self.cluster_map.shape, 3)
            plt.imshow(color_map)
        elif color_scheme=='classification':
            colors = (pltcolors.to_rgba_array(self.cluster_colors)*255).astype('uint8')[:,0:3]
            color_map = colors[self.cluster_map.reshape(-1)].reshape(*self.cluster_map.shape,3)
            # color_map = colors[self.cluster_map.reshape(-1)].reshape(*self.cluster_map.shape)
            # color_map = pltcolors.to_rgba_array(color_map)
            plt.imshow(color_map)
        elif color_scheme=='heatmap':
            #Simply use the cluster indexes as heatmap values for an easy way to make an image.
            plt.imshow(self.cluster_map)
        else:
            print('Invalid color scheme.')

    def create_cluster_colors(self, merged_csv_path, tol=5):
        """
        Uses merged cluster data to assign a color to each cluster based on association with a particular
          classification. Produces an array of colors that can be employed in plot_cluster_map.
        """
        #Load data containing cluster populations and classification:
        df = pd.read_csv(merged_csv_path)
        df = df.drop(['X','Y','pxlX','pxlY'],axis=1)
        #Find the mean cluster populations within each classification:
        farm_m = df.loc[(df['classification']=='farmland')].mean().values
        sett_m = df.loc[(df['classification']=='settlement')].mean().values
        wild_m = df.loc[(df['classification']=='wilderness')].mean().values

        """
        Assign a color to each cluster based on its relation to the 3 classes:
        b : blue. = settlement
        g : green. = wilderness
        r : red. = farmland
        c : cyan. = b+g
        m : magenta. = r+b
        y : yellow. = r+g
        k : black. = 0
        w : white. = all classes
        """
        colors = np.zeros(farm_m.size,dtype=object)
        for c in range(farm_m.size):
            if farm_m[c] > tol*np.max([sett_m[c],wild_m[c]]):
                colors[c] = 'r'
            elif sett_m[c] > tol*np.max([farm_m[c],wild_m[c]]):
                colors[c] = 'b'
            elif wild_m[c] > tol*np.max([sett_m[c],farm_m[c]]):
                colors[c] = 'g'
            elif farm_m[c]*tol < np.min([sett_m[c],wild_m[c]]):
                colors[c] = 'c'
            elif sett_m[c]*tol < np.min([farm_m[c],wild_m[c]]):
                colors[c] = 'y'
            elif wild_m[c]*tol < np.min([farm_m[c],sett_m[c]]):
                colors[c] = 'm'
            else:
                colors[c] = 'w'
        self.cluster_colors = np.append(['k'],colors) #adds an entry for the 0th cluster (i.e. unprocessed pixels)

foo = ApplyModel()

# #Load and plot the class grid over the TCI image:
# foo.load_class_grid('./USAID_africa/final_results/RFC100_Birch250_20180830/class_grid.npy')
# foo.plot_class_grid()

#Produce the cluster map with a class grid overlay:
# foo.compose_cluster_map('./USAID_africa/final_results/RFC100_Birch250_20180830/')
foo.load_cluster_map('./USAID_africa/final_results/RFC100_Birch250_20180830/cluster_map.npy')
foo.load_class_grid('./USAID_africa/final_results/RFC100_Birch250_20180830/class_grid.npy')
foo.create_cluster_colors('./USAID_africa/premerged/merged_csvs/merged_rbt5_with_birch_n250_rbt5_tr120pct.csv',
                          tol=5)
foo.plot_cluster_map(color_scheme='classification')
foo.overlay_class_grid()

#Zoom to a square area:
ax = plt.gca()
ax.set_xlim(8000,9400)
ax.set_ylim(3400,2000)
# plt.canvas.draw()

mng = plt.get_current_fig_manager()
mng.window.state('zoomed')

print('Done')

"""
Takes in the output file from usaid_search_esa.py and downloads ESA datasets.

Each dataset on average encompasses about 207 USAID points and is ~750MB in size.
"""
from satellite_head import *
import pickle
import zipfile
from timeit import default_timer as timer

OUTDIR = "./USAID_africa/ESA/"

#Load the list of good datasets:
filename = './usaid_selected_images_struct.pkl'
with open(filename,'rb') as pfile:
    master_struct = pickle.load(pfile)

sizes = [x['MBsizes'] for x in master_struct]
#Download datasets one-by-one:
for dataset in master_struct:
    #Make sure we don't already have this one:
    filename = dataset['titles']
    if not os.path.exists(OUTDIR + filename + ".SAFE"):
        print("\nTarget dataset:")
        print(dataset['titles'])
        print("Contains {} USAID points".format(dataset['points_inside']))
        #Download the set to the OUTDIR:
        t1 = timer()
        try:
            download_success = download_set(dataset['uuids'],filename+'.zip', outpath=OUTDIR)
            if download_success:
                print("Download took {:.1f} minutes".format((timer()-t1)/60))
                #Unzip the set:
                print("Unzipping dataset...")
                zip_ref = zipfile.ZipFile(OUTDIR+filename+'.zip', 'r')
                zip_ref.extractall(OUTDIR)
                zip_ref.close()
                #Delete the zip file:
                print("Removing temporary zip file...")
                os.remove(OUTDIR+filename+'.zip')
            else:
                print('DOWNLOAD FAILED.')
        except:
            print('DOWNLOAD FAILED.')

    else:
        print('---Already have dataset {}'.format(filename))

print('done')

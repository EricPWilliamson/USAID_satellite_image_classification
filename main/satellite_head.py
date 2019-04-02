###Look at satellite images from Sentinel 2's MSI instrument. Then modify and combine bands.
import time
import numpy as np
import matplotlib.pyplot as plt
import glymur
import scipy
import scipy.misc
import pandas
import xml.etree.ElementTree as ET
import pyproj
import requests
import re
import sys
import os
import fnmatch
import json
from my_utils import my_alert2 as myalert
from my_utils import imshow_to_file


def str2array(string):
    ###Take any string and pull out all the numbers. Works like sscanf.
    #Find any kind of integer or float in the string:
    numb_strings = re.findall(r'[-+]?[0-9]*\.?[0-9]+', string)
    #Convert each match into a float and load it into array:
    array = []
    for s in numb_strings:
        array += [float(s)]
    return array

def toClipboardForExcel(array):
    """
    Copies an array into a string format acceptable by Excel.
    Columns separated by \t, rows separated by \n
    """
    # Create string from array
    line_strings = []
    for line in array:
        line_strings.append("\t".join(line.astype(str)).replace("\n",""))
    array_string = "\r\n".join(line_strings)

    # Put string into clipboard (open, clear, set, close)
    clipboard.OpenClipboard()
    clipboard.EmptyClipboard()
    clipboard.SetClipboardText(array_string)
    clipboard.CloseClipboard()

def get_band(band_suffix, data_dir=None):
    ###Get the raw pixel values for an entire band
    if data_dir is None:
        IMG_DIR = "C:/Users/Rahakami/Downloads/S2A_MSIL1C_20170831T185921_N0205_R013_T10SEH_20170831T191016.SAFE/GRANULE/L1C_T10SEH_A011450_20170831T191016/IMG_DATA/"
        BASE_FN = "T10SEH_20170831T185921_"
        jp2_path = IMG_DIR + BASE_FN + band_suffix + ".jp2"
    else:
        search_pattern = "*_" + band_suffix + ".jp2"
        jp2_path = find_file(search_pattern, data_dir)
    jp2 = glymur.Jp2k(jp2_path)
    # thumbnail = jp2[::2, ::2]#This just skips every 2nd pixel.
    return jp2[:]

def scale_down(full_img, scale_factor=13.8278):
    """Converts all the values to uint8 format for proper RGB image display"""
    #Default scale factor is that used by TCI file
    int_img = np.round(full_img / scale_factor)

    sat_count = (int_img > 255).sum()
    tot_count = (int_img > 0).sum()
    if sat_count > 0:
        print("When rescaling, {} saturated values were lost ({:.2f}%).".format(sat_count, sat_count*100/tot_count))

    int_img = np.where(int_img > 255, 255, int_img)  # Hammer down anything above 255
    int_img = int_img.astype(np.uint8) #Converting type to unit8 is important, plt.imshow won't work properly without it
    return int_img

def latlon_to_pixel(xIn, yIn, inverse=False, data_dir=None):
    ###Converts pixel indices into lat-lon, or vice-versa.
    # Always assumes 10m resoltion.
    # Pixel indices are always treated as floats, and aren't rounded.

    #Get NW corner's UTM coords and the UTM zone from xml:
    if data_dir is None:
        xml_path = "C:/Users/Rahakami/Downloads/S2A_MSIL1C_20170831T185921_N0205_R013_T10SEH_20170831T191016.SAFE/GRANULE/L1C_T10SEH_A011450_20170831T191016/MTD_TL.xml"
    else:
        xml_path = find_file('MTD_TL.xml', data_dir)
    root = ET.parse(xml_path).getroot()
    geocoding = list(root.iter('Tile_Geocoding'))[0]
    epsg_code = geocoding.find('HORIZONTAL_CS_CODE').text #Geographical grid code, allows pyproj to convert coords.
    geopostion = geocoding.find('Geoposition')
    xNW = int(geopostion.find("ULX").text) #x position of NW corner, in meters
    yNW = int(geopostion.find("ULY").text) #y position of NW corner, in meters

    #Use pyproj to convert:
    myProj = pyproj.Proj(init=epsg_code)
    if inverse:
        utmX = xNW + xIn * 10  # Increasing as we go east.
        utmY = yNW - yIn * 10  # Decreasing as we go south.
        lon, lat = myProj(utmX, utmY, inverse=True)
        return lon,lat
    else:
        utmX, utmY = myProj(xIn, yIn, inverse=False)
        pxlX = (utmX-xNW)/10
        pxlY = -(utmY-yNW)/10
        return pxlX,pxlY

def dataset_search(search_query, start_row=0, num_rows=100, sort_config="&orderby=beginposition desc"):
    ###Searches the apihub database and parses search results.
    #Usage: dataset_search("q=filename:S2A* AND footprint:\"Intersects(38.000, -122.7000)\"")
    BASE_URL = "https://scihub.copernicus.eu/apihub/search?"
    D_NS = "{http://www.w3.org/2005/Atom}" #Default namespace in xml file
    OS_NS = "{http://a9.com/-/spec/opensearch/1.1/}" #Open Search namespace in xml file

    user, password = get_scihub_login()

    if num_rows>100:
        print("WARNING: Server won't allow over 100 rows.")
    row_config = "start={start_row}&rows={num_rows}&".format(start_row=start_row, num_rows=num_rows)

    server_command = BASE_URL + row_config + search_query + sort_config
    print("Contacting API HUB server...")
    r = requests.get(server_command, auth=(user, password))

    ###The server will return results as a string representing an xml file. Parse it:
    root = ET.fromstring(r.text)

    print(root[1].text)  # Show server response msg

    #Check the total number of results:
    tot_results = int(root.find(OS_NS + "totalResults").text)

    # Parse search results:
    titles = []
    cloud_covers = []
    datetimes = []
    MBsizes = []
    footprints = []
    uuids = []
    for search_result in root.iter(D_NS + 'entry'):
        # Get the name of each data set:
        # print(search_result.find(D_NS + "title").text)
        titles += [search_result.find(D_NS + "title").text]

        # Get the cloud cover percentages:
        s = search_result.find("*[@name='cloudcoverpercentage']")
        cloud_covers += [float(s.text)]
        # # Not all instruments have cloud cover values, enter -1 if none is present:
        # if type(s) is ET.Element:
        #     cloud_covers += [s.text]
        # else:
        #     cloud_covers += ['-1']

        #Get the datetime of the measurement:
        # s = search_result.find("*[@name='datatakesensingstart']") #This name is only used on datasets created after 2017-05-05
        s = search_result.find("*[@name='beginposition']") #This name seems to be used on all datasets
        # if type(s) is ET.Element:
        #     datestr += [s.text]
        # else:
        #     datestr += ['-1']
        #Parse date:
        datestr = s.text[0:19] #19 chars includes everything up to the ".###Z"
        # datetimes += [time.mktime(time.strptime(datestr, "%Y-%m-%dT%H:%M:%S"))]
        datetimes += [time.strptime(datestr, "%Y-%m-%dT%H:%M:%S")]


        #Get size of data set expressed in MB:
        # MBsizes += [search_result.find("*[@name='size']").text]
        sizestr = search_result.find("*[@name='size']").text
        if sizestr[-2:] == 'MB':
            MBsizes += [float(sizestr[:-3])]
        elif sizestr[-2:] == 'GB':
            MBsizes += [float(sizestr[:-3]) * 1024]
        else: #AFAIK the sizes are always in MB or GB...
            MBsizes += [-1]

        #Get coordinate pairs that make up the global footprint
        fpstr = search_result.find("*[@name='footprint']").text
        #Example string:
        #POLYGON ((-122.2627229165279 38.845005577616604,-122.29324528546964 38.75686232907166))
        fpstr = fpstr.split("(")[2] #Cut off beginning
        fpstr = fpstr.split(")")[0] #Cut off end
        fpstr = fpstr.split(",") #Separate into pairs
        fpmat = []
        for item in fpstr:
            subl = []
            for num in item.split(' '):
                subl.append(float(num))
            fpmat.append(subl)
        footprints.append(fpmat)

        #Get UUID in case we want to download the dataset:
        # uuids += [search_result.find("*[@name='uuid']").text] #Not every search result has this...wtf
        uuids += [search_result.find(D_NS + "id").text]

    results = {'titles': titles,
            'cloud_covers': cloud_covers,
            'datetimes': datetimes,
            'MBsizes': MBsizes,
            'footprints': footprints,
            'uuids': uuids}
    xml_text = r.text
    return results, tot_results, xml_text

def dataset_aggregate_search(search_query, max_results=1000):
    """Takes a search query and returns as many results as possible. Not limited by the 100 results per page limit."""
    results,n_results,xml_text = dataset_search(search_query,start_row=0)
    if n_results == 0:
        raise Exception("Error: No results found for query '{}'".format(search_query))
    for start_row in range(100,min([n_results,max_results]),100):
        new_results,new_n,xml_text = dataset_search(search_query,start_row=start_row)
        results = concdictlists(results,new_results)
    return results

def concdictlists(dict1, dict2):
    ###Concatenate all keys in two dictionaries. The dicts must have the same set of keys and each key must be a list.
    #Verify that keys are the same:
    if dict1.keys() != dict2.keys():
        print('Error: Dictionary keys do not match.')
        return {}
    #Concatenate:
    ds = [dict1, dict2]
    d = {}
    for k in dict1.keys():
        d[k] = dict1[k] + dict2[k]
        # d[k] = [d[k] for d in ds]
    return d

def restructure_results(results):
    """Repackages the results from a dataset_search into a list of dicts (akin to a Matlab struct)."""
    #Unpack results dictionary:
    titles = results['titles']
    datetimes = results['datetimes']
    cloud_covers = results['cloud_covers']
    MBsizes = results['MBsizes']
    footprints = results['footprints']
    uuids = results['uuids']
    #Load vars into a list of dicts:
    mystruct = []
    for i in range(len(MBsizes)):
        item = {'titles':titles[i],
                'cloud_covers':cloud_covers[i],
                'datetimes':datetimes[i],
                'MBsizes':MBsizes[i],
                'footprints':footprints[i],
                'uuids':uuids[i]}
        mystruct += [item]
    return mystruct

def assemble_query(coords, date_range='', max_clouds=100, exclude_old=False):
    ###Assemble a search string for use in dataset_search.
    #usage: date_range = "2017-01-01T00:00:00.000Z TO 2017-05-01T00:00:00.000Z"
    #If coords is a 2 column matrix with multiple locations, the search will require that all locations are included in the image.

    # Start with basic name search. We assume this is always desired:
    search_string = "q=filename:S2A*"
    # Add required footprint points:
    if type(coords) != 'numpy.ndarray':
        coords = np.array(coords)
    if coords.ndim == 2:
        for i in range(coords.shape[0]):
            lat = coords[i][1]
            lon = coords[i][0]
            search_string += " AND footprint:\"Intersects({:f}, {:f})\"".format(lat, lon)
    else:
        search_string += " AND footprint:\"Intersects({:f}, {:f})\"".format(coords[1], coords[0])
    # If desired, restrict date range:
    if len(date_range) > 0:
        search_string += " AND beginposition:[{}]".format(date_range)
    # If desired, add cloud cover restriction:
    if max_clouds < 100:
        search_string += " AND cloudcoverpercentage:[0 TO {:d}]".format(max_clouds)
    # If desired, exclude old data format datasets:
    if exclude_old:
        search_string += " NOT filename:*OPER_PRD_*"
    return search_string

def download_set(uuid, filename, outpath="./S2A_data/"):
    ###Download a dataset.
    SRU = "https://scihub.copernicus.eu/apihub/odata/v1/" #SERVICE_ROOT_URI
    D_NS = "{http://www.w3.org/2005/Atom}"  # Default namespace in xml file
    command_base = SRU + "Products('" + uuid + "')"
    success = False
    user, password = get_scihub_login()

    # # Get name of dataset:
    # server_command = command_base + "/Nodes"
    # r = requests.get(server_command, auth=(user, password))
    # root = ET.fromstring(r.text)
    # prod_name = root.find(D_NS + "entry/" + D_NS + "title").text

    #Clean up filename:
    if filename[-4:] != '.zip':
        filename += '.zip'

    save_path = outpath + filename
    server_command = command_base + "/$value"
    r = requests.get(server_command, auth=(user, password), stream=True)
    if r.status_code == 200:
        print("Downloading dataset...")
        with open(save_path, 'wb') as f:
            total_length = r.headers.get('content-length')
            sizeMB = int(total_length)/2**20 #1MB = 2**20 octets
            completed_bytes = 0
            #Download in 1MB chunks:
            for chunk in r.iter_content(1024*1024*1):
                completed_bytes += len(chunk)
                f.write(chunk)
                #Show download progress (does this waste significant resources?)
                sys.stdout.write("\r{:.1f}MB of {:.1f}MB".format(completed_bytes/2**20, sizeMB))
                sys.stdout.flush()
        if completed_bytes/2**20 == sizeMB:
            success = True
            print("\nDownload complete.")
        else:
            print("\nDownload incomplete.")
    else:
        print("Download error: code {:d}".format(r.status_code))
    return success

def find_file(pattern,path):
    """Finds a file matching a search pattern within a directory. Throws an error if no files were found, or multiple files were found."""
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    #Check that we only got one result:
    if len(result) != 1:
        raise Exception("Problem: found {} matches for '{}'".format(len(result), pattern))
    return result[0]

def point_in_footprint(footprint,coords):
    """Check each point for whether it lies within footprint:"""
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon

    fp_poly = Polygon(footprint)
    output = []
    for pair in coords:
        output += [fp_poly.contains(Point(pair))]
    return output

def check_footprint(data_dir, coords):
    """Determine if the points in coords fall within the footprint of this dataset. Returns a list of booleans corresponding to coords."""
    #Get global footprint from xml file:
    mtd_path = find_file("MTD_MSIL1C.xml",data_dir)
    tree = ET.parse(mtd_path)
    root = tree.getroot()
    footprint = list(root.iter('Global_Footprint'))[0]
    footprint_str = footprint.find('EXT_POS_LIST').text
    fp_coords = [float(s) for s in footprint_str.split()] #Convert text into floats
    fp_coords = np.flip(np.reshape(fp_coords,(-1,2)),axis=1) #Arrange in 2 columns of unspecified height
    #Check each point for whether it lies within footprint:
    return point_in_footprint(fp_coords,coords)

def box_in_footprint(data_dir):
    """Finds a rectangular box within an image that is fully covered by the footprint. The box is defined by pixel values x0,x1,y0,y1"""
    #Get global footprint from xml file:
    mtd_path = find_file("MTD_MSIL1C.xml",data_dir)
    tree = ET.parse(mtd_path)
    root = tree.getroot()
    footprint = list(root.iter('Global_Footprint'))[0]
    footprint_str = footprint.find('EXT_POS_LIST').text
    fp_coords = [float(s) for s in footprint_str.split()] #Convert text into floats
    fp_coords = np.flip(np.reshape(fp_coords,(-1,2)),axis=1) #Arrange in 2 columns of unspecified height
    #Convert the footprint to pixel values:
    pxl_fp = []
    for pair in fp_coords:
        x,y = latlon_to_pixel(pair[0],pair[1],inverse=False,data_dir=data_dir)
        pxl_fp += [[x,y]]
    #Try to find boundry levels that fit inside this footprint:
    x0 = int(min(p[0] for p in pxl_fp)) + 1
    x1 = int(max(p[0] for p in pxl_fp)) - 1
    y0 = int(min(p[1] for p in pxl_fp)) + 1
    y1 = int(max(p[1] for p in pxl_fp)) - 1
    corners = point_in_footprint(pxl_fp,[[x0,y0], [x0,y1], [x1,y1], [x1,y0]])
    while False in corners:
        n_x0 = x0
        n_y0 = y0
        n_x1 = x1
        n_y1 = y1
        if not corners[0]:
            n_x0 = x0+10
            n_y0 = y0+10
        if not corners[1]:
            n_x0 = x0+10
            n_y1 = y1-10
        if not corners[2]:
            n_x1 = x1-10
            n_y1 = y1-10
        if not corners[3]:
            n_x1 = x1-10
            n_y0 = y0+10
        x0 = n_x0
        y0 = n_y0
        x1 = n_x1
        y1 = n_y1
        corners = point_in_footprint(pxl_fp,[[x0,y0],[x0,y1],[x1,y1],[x1,y0]])
    return [x0,x1,y0,y1]

def get_scihub_login(cred_filepath='./scihub_credentials.json'):
    """Loads username and password for scihub from a json file."""
    with open(cred_filepath) as json_data_file:
        credential_dict = json.load(json_data_file)
    return credential_dict['user'], credential_dict['pass']

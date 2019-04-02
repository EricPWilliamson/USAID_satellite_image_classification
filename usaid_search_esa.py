"""
Systematically selects Sentinel-2 data corresponding to areas covered by the USAID data set. The resulting list is
  saved so that we can later download each dataset one-by-one.

First searches for datasets containing a USAID point, then checks those results for an image containing at least 100
  USAID points, so that we dont waste time downloading images for only a few points. Tries to find images without
  clouds close to Dec2017.

A grid-by-grid search might be more efficient, but we just select search points from the USAID set at random. This still
  works well enough.

A search on API Hub takes about 2 seconds, so it's more efficient to avoid doing too many. Searching is the primary
  bottleneck here.

We normally get about 200 matching points per satellite image, so accounting for the bulk of the 26000+ USAID datapoints
  will take about 130 searches or ~5mins, but searching at outlying points could take a long time. We will limit the
  search to 10 minutes.
"""
from satellite_head import *
import pandas as pd
from timeit import default_timer as timer
import pickle


###Load tidy USAID data:
master_df = pd.read_csv('./USAID_africa/tidy_data.csv')
usaid_coords = master_df[['X','Y']].values

###Repeatedly search the database for locations in USAID database, removing points as they are taken in:
t0 = timer()
master_struct = []
while (usaid_coords.shape[0] > 1000) and (timer()-t0 < 600):
    print("Searching with {} points remaining...".format(usaid_coords.shape[0]))
    found_image = False
    #Randomly pick a point from USAID:
    search_idx = np.random.randint(0, usaid_coords.shape[0])
    search_coord = usaid_coords[search_idx]

    #Search around the coordinate point, for dates in Nov'17 to Jan'18, with max 2% cloud cover:
    search_query = assemble_query(search_coord, date_range="2017-11-01T00:00:00.000Z TO 2018-02-01T00:00:00.000Z", max_clouds=2, exclude_old=True)

    #See all results for this point:
    results = dataset_aggregate_search(search_query)
    #Reorganize results dictionary:
    datetimes = results['datetimes']
    result_struct = restructure_results(results)

    #Favor images close to 12/25/2017
    ideal_time = time.strptime('2017-12-25', "%Y-%m-%d")
    time_errors = []
    for dt in datetimes:
        time_errors += [np.abs(time.mktime(dt) - time.mktime(ideal_time))]
    sort_idx = np.argsort(time_errors)

    ###Go through results in our sorted order until we find one that meets our criteria:
    for i in sort_idx:
        is_inside = point_in_footprint(result_struct[i]['footprints'],usaid_coords)
        points_inside = sum(1 for i in is_inside if i)
        if points_inside > 100:
            found_image = True
            print("Selected ESA image containing {} USAID datapoints.".format(points_inside))
            #Add this result to master_struct:
            new_entry = result_struct[i]
            new_entry['points_inside'] = points_inside
            master_struct += [new_entry]
            #Remove the USAID points that were just accounted for.
            usaid_coords = usaid_coords[np.invert(is_inside)]
            break
    #If no good image was found, just forget about this search point. It is probably outside of the main region:
    if not found_image:
        print("No adequate images for this search point.")
        usaid_coords = np.delete(usaid_coords, search_idx, axis=0)

###Save the struct of selected ESA images:
filename = 'usaid_selected_images_struct.pkl'
with open(filename, 'wb') as pfile:
    pickle.dump(master_struct, pfile, pickle.HIGHEST_PROTOCOL)

print('Done')

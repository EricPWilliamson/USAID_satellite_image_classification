"""
Import and organize data from geo-wiki's USAID data set, comprising of information on sites in Africa.

Each site can be classified as settlement or cultivation (or none). The classification is scored on a 4 point scale, presumably
  indicating the degree of cultivation/settlement. The data was crowd sourced from volunteers who examined satellite
  images, so erroneous data points can be expected. User ID was also tracked, potentially allowing us to flag users for
  erroneous entries. Users also entered a confidence score for each entry--another potential way to eliminate outliers.

The sites are tagged with a single lat,lon coordinate, which can be mapped to a coordinate in the Sentinel-2 dataset.

Data was entered on September 14-16, 2012.

Data contains 77480 entries with no missing values.

!!!Many points contain duplicate input from multiple users. They have the same PointID value. Look for agreement and combine user input.
"""

import pandas as pd
import csv
import matplotlib.pyplot as plt

pd.set_option('display.expand_frame_repr', False)


def google_maps_export(filename,coords):
    """Export a list of lon,lat coordinates to csv in a format that can be imported into Google Maps. Google allows 2000 rows max."""
    fullpath = './USAID_africa/'+filename+'.csv'
    with open(fullpath,'w',newline="\n") as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(['Positions','Names'])
        for i,pair in enumerate(coords):
            spamwriter.writerow(["{:.15f} {:.15f}".format(pair[0],pair[1]),"P{}".format(i)])
    print('{}.csv saved.'.format(filename))

def export_area(df):
    ###Find the geographical boundries of the data:
    # #Simple rectangle:
    # minx = df['X'].min(axis=0)
    # maxx = df['X'].max(axis=0)
    # miny = df['Y'].min(axis=0)
    # maxy = df['Y'].max(axis=0)
    # boundry_box = [[minx,maxy],[maxx,maxy],[maxx,miny],[minx,miny]]

    # #Actual boundries:
    # minx = df.loc[df['X'].idxmin()][['X','Y']].values.tolist()
    # maxx = df.loc[df['X'].idxmax()][['X','Y']].values.tolist()
    # miny = df.loc[df['Y'].idxmin()][['X','Y']].values.tolist()
    # maxy = df.loc[df['Y'].idxmax()][['X','Y']].values.tolist()
    # boundry_box = [minx,maxx,miny,maxy]

    #Random sampling of points:
    boundry_box = df.sample(n=1000)[['X','Y']].values.tolist()

    ###Save csv file in the format Google Maps accepts:
    google_maps_export('data_boundries',boundry_box)

def export_classifications(df):
    """Export a file for each classification, so that they can later be distinguished in Google Maps"""
    for desc in df.classification.unique():
        subdf = df[df.classification == desc]
        if subdf.shape[0] > 2000:
            subdf = subdf.sample(n=2000)
        google_maps_export(desc+'_points', subdf[['X','Y']].values.tolist())

def pie_charts(df):
    """Show relative occurence rates of the different classes."""
    plt.ion()

    plt.figure()
    counts = master_df.PointID.groupby(master_df.CULTIVATION).count()
    plt.axis('equal')
    plt.pie(counts,labels=counts.index, autopct='%.2f%%')
    plt.title('Cultivation value distribution')

    plt.figure()
    counts = master_df.PointID.groupby(master_df.SETTLEMENT).count()
    plt.axis('equal')
    plt.pie(counts,labels=counts.index, autopct='%.2f%%')
    plt.title('settlement value distribution')

    # plt.figure()
    # counts = master_df.PointID.groupby(master_df.classification).count()
    # plt.axis('equal')
    # plt.pie(counts,labels=counts.index)
    # plt.title('Classification distribution (set. , cul.)')

    class_pct = df.classification.value_counts() * 100/df.shape[0]
    plt.figure()
    plt.axis('equal')
    plt.pie(class_pct,labels=class_pct.index,autopct='%.2f%%')
    plt.title('Classification distribution')

def basic_cleanup(df):
    """Performs basic wrangling of the dataset: Removes ambiguous data points, assigns a simple classification to each data point."""
    ###Create a new feature combining SETTLEMENT and CULTIVATION:
    df['classification'] = df['SETTLEMENT'].astype(str)+','+df['CULTIVATION'].astype(str)

    ###Drop unwanted data points:
    #Points without high confidence are probably not worth using:
    df = df[df['CONFIDENCE'] == 3]
    #Since CONDIENCE is now uniform, we dont need this feature anymore:
    df = df.drop(['CONFIDENCE'],axis=1)
    #Points with barely any settlement/cultivation are kind of ambiguous. It's not clear to me whether the user saw mostly
    #  wilderness, or mostly human activity. It is best to just eliminate them.
    df = df[df['classification'] != '1,0']
    df = df[df['classification'] != '1,1']
    df = df[df['classification'] != '0,1']

    ###Simplify classification feature:
    #Reduce the classification feature into three simple categories:
    df['classification'] = df['classification'].replace('0,0','wilderness')
    df['classification'] = df['classification'].replace(regex=r'[01],[23]',value='farmland')
    df['classification'] = df['classification'].replace(regex=r'[23],.',value='settlement')

    return df

def compare_users(df):
    """
    Takes a particular point and checks for agreement between different users. Unless 90% of users agree, the point
      will be classified as 'unknown'. Returns just one row, since we want to consolidate.
    """
    if df.classification.value_counts().max() > 0.9*df.shape[0]:
        df.classification = df.classification.value_counts().idxmax()
        return df.iloc[0,:]
    else:
        df.classification = 'unknown'
        return df.iloc[0,:]

def consolidate_duplicates(df):
    """Use cross-comparison of users to consolidate and improve data accuracy. Here, we also discard the columns that won't be used in the final analysis."""
    #First remove any samples that were checked only once. These are not reliable.
    df = df.groupby("PointID").filter(lambda x: len(x) > 1)
    #Consolidate classifications made by multiple users, so that we only have one classification per point.
    df = df.groupby('PointID')[['X','Y','classification']].apply(compare_users)
    #Remove cases where users disagreed:
    df = df[df['classification'] != 'unknown']
    return df

master_df = pd.read_csv('./USAID_africa/user_data.csv')
master_df = basic_cleanup(master_df)
master_df = consolidate_duplicates(master_df)



###Examine dataset:
# master_df.head()
# master_df.info()
# master_df['CONFIDENCE'].describe()
# master_df.describe()
# master_df['USERID'].value_counts()
# export_area(master_df)
# export_classifications(master_df)
# pie_charts(master_df)

###Save to csv:
master_df.to_csv('./USAID_africa/tidy_data.csv',index=False)


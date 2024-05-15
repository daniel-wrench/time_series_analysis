# Identify the location of any missing data in and between the CDFs
# in the PSP and MMS directories

import data_import_funcs as data_import
import os
import pandas as pd
import numpy as np
import datetime as dt

################################################################################

resample_freq = '0.75S'
print("\nCHECKING PSP FILES \nfollowing resampling to " + resample_freq + "\n")

# Create the list of files (all those in the directory)
psp_list_saved = os.listdir("data_raw/psp")
psp_list_saved.remove('.gitkeep')
if 'gmon.out' in psp_list_saved:
    psp_list_saved.remove('gmon.out')
psp_list_saved_complete = ["data_raw/psp/" + i for i in psp_list_saved]
psp_list_saved_complete.sort()

# Identify gaps BETWEEN files, rather than within them
# defined as where the difference between adjacent timestamps is greater than twice the median difference

psp_timestamps_int = [int(i[32:42]) for i in psp_list_saved_complete]

psp_timestamps = [dt.datetime(
    int(i[32:36]), 
    int(i[36:38]), 
    int(i[38:40]), 
    int(i[40:42])) for i in psp_list_saved_complete]

psp_timestamps_array = np.array(psp_timestamps)
psp_timestamp_diffs = np.diff(psp_timestamps)
psp_timestamp_diffs = np.append(psp_timestamp_diffs, dt.timedelta(0))
# Getting array of those files that are followed by a gap
gaps_between_files = np.array(psp_timestamps_int)[psp_timestamp_diffs >
                                          2 * np.median(psp_timestamp_diffs)]


for i in range(len(psp_list_saved_complete)):
    data = data_import.read_cdfs([psp_list_saved_complete[i]],
                                 {'epoch_mag_RTN': (0), 'psp_fld_l2_mag_RTN': (0, 3), 'label_RTN': (0, 3)})

    psp_data_ready = data_import.extract_components(
        data, var_name='psp_fld_l2_mag_RTN', label_name='label_RTN', time_var='epoch_mag_RTN', dim=3)

    psp_df = pd.DataFrame(psp_data_ready)

    psp_df['Time'] = pd.to_datetime(
        '2000-01-01 12:00') + pd.to_timedelta(psp_df['epoch_mag_RTN'], unit='ns')
    psp_df = psp_df.drop(columns='epoch_mag_RTN').set_index('Time')

    if int(psp_list_saved_complete[i][32:42]) in gaps_between_files:
        print("GAP BETWEEN THIS INTERVAL AND FOLLOWING")

    # Original freq is 0.007s. Resampling to get appropriate number of correlation times in 10,000 points
    psp_df_resampled = psp_df.resample(resample_freq).mean()

    #print("Number of missing values after resampling:")
    # print(psp_df_resampled.isnull().sum())
    if psp_df_resampled.isnull().sum().any() > 0:
        print("DATA MISSING")
    print("\n")

################################################################################

resample_freq = '0.008S'
print("\nCHECKING MMS FILES \nfollowing resampling to " + resample_freq + "\n")

# Create the list of files (all those in the directory)
mms_list_saved = os.listdir("data_raw/mms")
mms_list_saved.remove('.gitkeep')
if 'gmon.out' in mms_list_saved:
    mms_list_saved.remove('gmon.out')
mms_list_saved_complete = ["data_raw/mms/" + i for i in mms_list_saved]
mms_list_saved_complete.sort()

# Identify gaps BETWEEN files, rather than within them
# defined as where the difference between adjacent timestamps is greater than twice the median difference

mms_timestamps_int = [int(i[30:44]) for i in mms_list_saved_complete]

mms_timestamps = [dt.datetime(
    int(i[30:34]), 
    int(i[34:36]), 
    int(i[36:38]), 
    int(i[38:40]),
    int(i[40:42]),
    int(i[42:44])) for i in mms_list_saved_complete]

mms_timestamps_array = np.array(mms_timestamps)
mms_timestamp_diffs = np.diff(mms_timestamps)
mms_timestamp_diffs = np.append(mms_timestamp_diffs, dt.timedelta(0))
# Getting array of those files that are followed by a gap
gaps_between_files = np.array(mms_timestamps_int)[mms_timestamp_diffs >
                                          1.5 * np.median(mms_timestamp_diffs)]                                          

for i in range(len(mms_list_saved_complete)):
    data = data_import.read_cdfs([mms_list_saved_complete[i]],
                                 {'Epoch': (0), 'mms1_fgm_b_dmpa_brst_l2': (0, 4), 'label_b_gse': (0, 4)})
    mms_data = data_import.extract_components(
        data, var_name='mms1_fgm_b_dmpa_brst_l2', label_name='label_b_gse', time_var='Epoch', dim=4)
    mms_df = pd.DataFrame(mms_data)
    mms_df['Time'] = pd.to_datetime(
        '2000-01-01 12:00') + pd.to_timedelta(mms_df['Epoch'], unit='ns')
    mms_df = mms_df.drop(columns='Epoch').set_index('Time')
    mms_df = mms_df.drop('Bt', axis=1)

    if int(mms_list_saved_complete[i][30:44]) in gaps_between_files:
        print("GAP BETWEEN THIS INTERVAL AND FOLLOWING")

    # Original freq is 0.007s. Resampling to get appropriate number of correlation times in 10,000 points
    mms_df_resampled = mms_df.resample(resample_freq).mean()

    #print("Number of missing values after resampling:")
    # print(mms_df_resampled.isnull().sum())
    if mms_df_resampled.isnull().sum().any() > 0:
        print("DATA MISSING")
    print("\n")

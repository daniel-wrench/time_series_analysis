import pandas as pd
import numpy as np
# from spacepy import pycdf
import cdflib
import matplotlib.pyplot as plt
import datetime
import math as m
import datetime

def read_cdfs(filelist, variables, mask=None):
   """
      Read a list of cdf files, and create a dictionary with their data
      combined in the same order that the files are listed.
      filelist: List of files to read 
      variables: Dictionary with the syntax: {'variable_name':tuple}
                 where tuple is (0) for a one dimensional time series, 
                 and (0,n) where n is the number of columns in the variable
                 
                 If all variables are simple one dimensional time series,
                 they can be passed simply as a list or tuple of strings:
                 e.g. ['Epoch','br','bt','bn'] etc.
      freq: sampling frequency of series to resample on
            e.g. '48S'
      mask: variable name and column to use for mask as a list or tuple
            e.g. for sweap on psp: mask=['DQF',0]
      returns a ditionary with variable names with corresponding data read
      from cdf files
      example:
         d=read_cdfs(['spp_swp_spc_l3i_20181031_v05.cdf',\
                      'spp_swp_spc_l3i_20181101_v07.cdf'],\
            {'vp_moment_RTN':(0,3),'np_moment':(0),'Epoch':(0)}
   """
#    from spacepy import pycdf

   if type(variables) in (list,tuple):
      vardict={}
      for vrbl in variables: vardict[vrbl]=(0)
   else:
      vardict=variables.copy()

#create empty numpy arrays
   dictionary={} 
   for j in vardict.keys():
      dictionary[j]=np.empty(vardict[j])

# read data from cdf files and append the arrays.
   for i in filelist:
      print('reading file ',i)
      # Open CDF file
      d = cdflib.CDF(i)
      # Read in the mask
      if mask is not None:
         if (type(mask) is str):
            mask=cdflib.varget(d[mask])
         elif (type(mask) in (list,tuple)):
            if len(mask) == 1:
               mask=cdflib.varget(d[mask[0]])
            else:
               mask=cdflib.varget(d[mask[0]])[:,mask[1]]
         mask=np.array(mask)
         dictionary['mask']=mask

      # Loop over all variables
      for j in vardict.keys():
         # Read j into a temporary buffer
         tmp=d.varget(j)
         # Set bad data to np.NaN: CAN'T GET THIS TO WORK - AND IS IT EVEN NECESSARY?
         # if 'FILLVAL' in d.varattsget(j):
         #    tmp[np.where(tmp==d.varattsget(j)['FILLVAL'])] = np.NaN
         # create mask appropriate fot the variable shape
         if mask is not None: 
            if len(tmp.shape) > 1:
               mmask=np.empty(tmp.shape)
               for i in range(tmp.shape[1]): mmask[:,i]=mask
            else:
               mmask=mask
            tmp=np.ma.masked_array(data=tmp,mask=mmask,fill_value=np.NaN)
         # append the dictionary
         dictionary[j]=np.append(dictionary[j], tmp,axis=0)

      #Working on trying to affix labels to multi-dimensional vars

      # for j in vardict.keys():
      #    if 'LABL_PTR_1' in d.varattsget(j):
      #       field_compon = {}
      #       comp_label_var = d.varattsget(j)['LABL_PTR_1']
      #       for x in range(4):
      #          y = dictionary[j][:,x]
      #          field_compon[dictionary[d.varattsget(j)][0,x].strip()] = y

   print('Done reading data')

   return dictionary




def date_1d_dict(dictionary, frequency = 0, convert = 0):
   """
      Reads a dictionary (created from a CDF file) and returns a Pandas DataFrame with a datetime index.
      Requires dictionary to have an Epoch key to then be converted into a datetime object.
      If frequency is specified, then data will be resampled accordingly
      If conversion = 1, then function will also calculate lambda and delta values from BT, BR, and BN components
   """
   data = pd.DataFrame(dictionary)
   time = data['Epoch'].values
   dates = cdflib.epochs.CDFepoch.to_datetime(time, to_np=True)
   datesframe = pd.DataFrame(dates)
   datesframe.columns = ['Time']
   data['Time'] = datesframe #Linking to variables
   data = data.set_index('Time').drop('Epoch', axis=1)

   #Calculating elevation and azimuthal angles from RTN components
   if 'F1' in data.columns and convert == 1:

      def convert_angle(x):
         if x < 0: x+=360
         return(x)

      data['lambda'] = data['BT'].div(data['BR']).apply(m.atan).apply(m.degrees).apply(convert_angle)
      data['delta'] = data['BN'].div(data['F1']).apply(m.asin).apply(m.degrees)

      #Note this alt method for calculating lambda which currently looks more similar:
      # Therefore I've opted for a different method, shown here
      #voyager_data['hypotenuse'] = (voyager_data['BT']**2).add(voyager_data['BR']**2).apply(m.sqrt)
      # Fixing issue where BT is somehow greater than hypotenuse
      #voyager_data['BT'] = np.where(abs(voyager_data['BT']) > abs(voyager_data['hypotenuse']), voyager_data['hypotenuse'], voyager_data['BT'])
      #voyager_data['lambda'] = voyager_data['BT'].div(voyager_data['hypotenuse']).apply(m.asin).apply(m.degrees)*(-1)+180
   
   print("Here is a snippet of the raw data (before re-sampling according to freq)")
   print(data.head())

   if frequency != 0:
      data = data.resample(frequency).mean()

   return data



def read_asc_ts(file, cols, vars_fillvals, freq):
   """
      Reads an ASC (txt) file and returns a Pandas DataFrame with a datetime index.
      file: ASC file
      cols: list of column indices to select
      vars_fillvals: dictionary of missing value replacements for each column name selected
      E.g. {'Year':999, 'Proton temperature': 9.999}
      freq: Sampling frequency, e.g. 'D'
   """
   raw_data = pd.read_table(file, 
                                sep = '\s+',
                                header = None,
                                usecols=cols,
                                na_values = vars_fillvals,
                                names = list(vars_fillvals.keys()))
                                #'Field Magnitude Average' (here = B) and 'Magnitude of Average Field' are very similar
                                # .dropna() to remove missing values

   raw_data['Time'] = pd.to_datetime(raw_data['Year'], format = "%Y") + pd.to_timedelta(raw_data['Day'] - 1 , unit = 'days') + pd.to_timedelta(raw_data['Hour'], unit = 'hours')
   print("The unique hours in this dataset are:", format(raw_data['Hour'].unique())) #All hours are 0 - as expected for this daily data

   data = raw_data.drop(columns = ['Year', 'Day', 'Hour']).set_index('Time').asfreq(freq)

   return data


def extract_components(dict, var_name, label_name, time_var, dim):
   field_compon = {}
   for x in range(dim):
      y = dict[var_name][:,x]
      field_compon[dict[label_name][0,x].strip()] = y
   field_compon[time_var] = dict[time_var]
   return field_compon

def pltpwrl(x0,y0,xi=1,xf=10,alpha=-1.66667,ax=None,**kwargs):
   """
      Plots a power-law with exponent alpha between the 
      xrange (xi,xf) such that it passes through x0,y0
   """
   import numpy as np
   import matplotlib.pyplot as plt
   x=np.linspace(xi,xf,50)
   if ax is None:
      ax=plt.gca()
   ax.plot(x,(y0*x0**-alpha)*x**alpha,**kwargs)
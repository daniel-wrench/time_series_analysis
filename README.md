# Gaps on structure functions: **interactive pipeline demo**
1. ~~Email Marcus~~
1. ~~Get heatmap importing and exporting~~
1. Plots
    - ~~MAPE in error trend line plots (means vs. means of means)? Using these in test set eval as well as determining bias initially?~~
    - ~~Add file indexing to later plots, move to appropriate locations.~~
    - ~~Get sample size plot (add *n* to heatmaps) Especially interested in how 3d heatmaps become more populated - *how often can test intervals not find the nearest bin?*~~
    - ~~Get plots to save. At earliest point in the code, whereever we're saving things, run the os func to check folder exists~~
1. ~~Run for 2 PSP ints for training, 1 PSP for testing.~~ See what dfs we are left with, and export the final ones.
1. ~~Run for 1 Wind testing~~
1. ~~Save and commit~~
2. ~~Move concatenation, as done as at start of step 3, from start of step 4 to start of step 5. Make step 4 run *on each test set file individually*.~~
1. ~~Divide each of the numbered steps into separate notebooks.~~ 
2. ~~Test these separate notebooks locally: 3 and 2 PSPs, 2 Wind~~
2. ~~Make quick to run on PSP vs. Wind, especially initial processing. Perhaps to just have if spacecraft =="psp": do this, elif "wind". Possibly easier to test this in scripts rather than notebooks.~~
2. ~~Quick run of each notebook~~
2. ~~Quick scroll through notebooks, checking for unnecessary outputs~~
3. ~~Convert notebooks to scripts.~~
    - ~~Note text output, make sure informative/concise enough.~~
    - ~~Turn file_index into a sys argument (this will be the job # on the HPC)~~
2. Create new repo locally with necessary files
5. Draft submission scripts, with job arrays for steps 1 and 3.
1. Push new repo to GitHub
2. Speed up Wind data reading
2. clone to NeSI.
9. Quick scaling study on NeSI, investigating bad ints and possible automatic removal during download or initial reading
10. Results for 1 year of PSP, 1 month of Wind
10. Up the ante to two years, **while writing up existing results**. Maybe set aside geostats stuff for now.
11. Send manuscript to Tulasi, Fraternale, Marcus
12. Implement Fraternale's sample size threshold for fitting slopes

#### Notes
- Can add smoothing to correction step alter, **not on critical path for getting most of the scripts on NESI**
- Having logarithmically spaced lag bins would make the correction factor much cleaner to work with: one-to-one bins
- For now likely to do stick with simple job arrays and single jobs on HPC, with importing and exporting of intermediate steps, but perhaps better to do single MPI script with broadcasting and reducing.
- If we tend to just get one 1 interval out of each file, could do away with (file,int) indexing
- In either case, might using actual pandas indexes be easier/more efficient?
- Add sf slope to Wind dataset

#### Download files

Run in terminal

```
# For selecting a range of dates

wget --no-clobber --directory-prefix=data/raw/psp  https://spdf.gsfc.nasa.gov/pub/data/psp/fields/l2/mag_rtn/2018/psp_fld_l2_mag_rtn_201811{0200..0318}_v02.cdf
wget --no-clobber --directory-prefix=data/raw/wind/ https://spdf.gsfc.nasa.gov/pub/data/wind/mfi/mfi_h2/2016/wi_h2_mfi_201601{01..07}_v05.cdf

# For entire folders

wget --no-clobber --directory-prefix=data/raw/psp --recursive -np -nv -nH --cut-dirs=7 --accept cdf  https://spdf.gsfc.nasa.gov/pub/data/psp/fields/l2/mag_rtn/2018/
wget --no-clobber --directory-prefix=data/raw/wind --recursive -np -nv -nH --cut-dirs=2 --accept cdf  https://spdf.gsfc.nasa.gov/pub/data/wind/mfi/mfi_h2/
```

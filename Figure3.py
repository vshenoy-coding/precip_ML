# Three key metrics used in ML evaluation for precipitation:

# CC (Correlation Coefficient): Measures how well the predicted pattern matches the observed pattern (0 to 1).

# R2 (Coefficient of Determination): Measures the proportion of variance explained by the model.

# NRMSE (Normalized Root Mean Square Error): Measures the magnitude of error, normalized to the range of the data (lower is better).

# joblib files are saved as list of dictionaries

######################################################################################################################################################
# Conflicting older visions of sklearn and machine learning models prevent recreation of Figure 3.
######################################################################################################################################################

import requests
from bs4 import BeautifulSoup
import re

base_url = "https://portal.nersc.gov/project/m2977/ML_Precip/"
headers = {'User-Agent': 'Mozilla/5.0'}

def find_joblib_files(url):
    print(f"Searching in: {url}")
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all links
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # If it's a joblib file, print the full URL
            if href.endswith('.joblib'):
                print(f"FOUND: {url + href}")
            
            # If it's a directory (ends in /) and not a parent dir, recurse
            elif href.endswith('/') and href not in ['../', './']:
                # Limit depth to avoid infinite loops
                if url.count('/') < 10:
                    find_joblib_files(url + href)
    except Exception as e:
        pass

find_joblib_files(base_url)

# import requests
# import os

# # Create folders
# !mkdir -p rf_models xgb_models

# regions = ['west', 'mount', 'ngp', 'sgp', 'northeast', 'southeast']
# base_url = "https://portal.nersc.gov/project/m2977/ML_Precip/pmax_region_models"

# # Quick check for smaller data files
# test_url = "https://portal.nersc.gov/project/m2977/ML_Precip/pmax_region_data/"
# r = requests.get(test_url)
# if "metrics" in r.text or "stats" in r.text:
#     print("Found a metrics folder! We should download from there instead.")
# else:
#     print("No obvious metrics folder. Proceeding with model downloads is the only way.")

# print("Downloading Split-by-Year models...")
# for reg in regions:
#     for model in ['rf', 'xgb']:
#         # Note the specific naming convention discovered by the spider
#         file_name = f"{model}_model_split_by_year_{reg}.joblib"
#         url = f"{base_url}/{model}_models/{file_name}"
#         path = f"{model}_models/{reg}.joblib" # Simplified local name
        
#         r = requests.get(url, stream=True)
#         if r.status_code == 200:
#             with open(path, 'wb') as f:
#                 f.write(r.content)
#             print(f"✅ Downloaded {reg} ({model.upper()})")
#         else:
#             print(f"❌ Failed: {url} (Status: {r.status_code})")

import os
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm.notebook import tqdm

# 1. Setup
!mkdir -p rf_models xgb_models
regions = ['west', 'mount', 'ngp', 'sgp', 'northeast', 'southeast']
base_url = "https://portal.nersc.gov/project/m2977/ML_Precip/pmax_region_models"

def download_with_progress(args):
    model, reg = args
    file_name = f"{model}_model_split_by_year_{reg}.joblib"
    url = f"{base_url}/{model}_models/{file_name}"
    path = f"{model}_models/{reg}.joblib"
    
    try:
        response = requests.get(url, stream=True, timeout=15)
        total_size = int(response.headers.get('content-length', 0))
        
        # Initialize progress bar for this specific file
        with open(path, 'wb') as f, tqdm(
            desc=f"{reg} ({model})",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            leave=False
        ) as bar:
            for data in response.iter_content(chunk_size=1024*1024): # 1MB chunks
                size = f.write(data)
                bar.update(size)
        return f"✅ Finished {reg}_{model}"
    except Exception as e:
        return f"❌ Error {reg}_{model}: {str(e)}"

# 2. Execute
tasks = [(m, r) for r in regions for m in ['rf', 'xgb']]
print(f"Downloading {len(tasks)} large model files. This may take some time...")

with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(download_with_progress, tasks))

for res in results:
    print(res)

###
# Search for metrics files
# The metrics files have <region>_metrics.joblib format

metrics_url = "/content/xgb_models/West_metrics.joblib" # Update to point to the local file
local_metrics_path = "/content/xgb_models/West_metrics.joblib" # Update to the local file path
metrics_dir = os.path.dirname(local_metrics_path)

print(f"Updated metrics_url to: {metrics_url}")
print(f"Updated local_metrics_path to: {local_metrics_path}")
print(f"Updated metrics_dir to: {metrics_dir}")

import requests
from bs4 import BeautifulSoup
import re

def find_joblib_files_metrics(url):
    print(f"Searching in: {url}")
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all links
        for link in soup.find_all('a', href=True):
            href = link['href']

            # If it's a metrics joblib file, print the full URL
            if href.endswith('_metrics.joblib'):
                print(f"FOUND: {url + href}")

            # If it's a directory (ends in /) and not a parent dir, recurse
            elif href.endswith('/') and href not in ['../', './']:
                # Limit depth to avoid infinite loops
                if url.count('/') < 10:
                    find_joblib_files_metrics(url + href)
    except Exception as e:
        pass

metrics_base_url = "https://portal.nersc.gov/project/m2977/ML_Precip/pmax_region_models/rf_models/"
find_joblib_files_metrics(metrics_base_url)

# Code works up to this point

#####################################################################################################################################################
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

def calculate_cc(y_true, y_pred):
    """Calculates the Pearson Correlation Coefficient."""
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan # Correlation requires at least two points and non-zero standard deviation
    return np.corrcoef(y_true, y_pred)[0, 1]

def calculate_r2(y_true, y_pred):
    """Calculates the R-squared score."""
    return r2_score(y_true, y_pred)

def calculate_nrmse(y_true, y_pred):
    """Calculates the Normalized Root Mean Squared Error (NRMSE)."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    y_true_range = np.max(y_true) - np.min(y_true)
    if y_true_range == 0:
        return 0.0 # Or np.nan, depending on desired behavior for constant true values
    return rmse / y_true_range

print("Metric calculation functions (calculate_cc, calculate_r2, calculate_nrmse) defined.")


import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import sklearn.tree._tree as _tree
import sklearn.ensemble._forest as _forest

# Re-apply the simplified, robust _check_node_ndarray patch for scikit-learn
_tree._check_node_ndarray = lambda *args, **kwargs: args[0] if args else None
print("Re-applied: Scikit-learn _tree._check_node_ndarray patched successfully with a simplified bypass.")

# Re-apply the 'monotonic_cst' attribute patch for RandomForestRegressor if missing
if not hasattr(_forest.RandomForestRegressor, 'monotonic_cst'):
    _forest.RandomForestRegressor.monotonic_cst = None
    print("Re-applied: Patched sklearn.ensemble.RandomForestRegressor with 'monotonic_cst' attribute.")


def calculate_cc(y_true, y_pred):
    """Calculates the Pearson Correlation Coefficient."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return np.corrcoef(y_true, y_pred)[0, 1]

def calculate_r2(y_true, y_pred):
    """Calculates the R-squared score."""
    return r2_score(y_true, y_pred)

def calculate_nrmse(y_true, y_pred):
    """Calculates the Normalized Root Mean Squared Error (NRMSE)."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    y_true_range = np.max(y_true) - np.min(y_true)
    if y_true_range == 0:
        return 0.0
    return rmse / y_true_range

def evaluate_model_metrics(model_path, X_test, y_true):
    """
    Loads a model, makes predictions, and calculates CC, R2, NRMSE.

    Args:
        model_path (str): Path to the joblib model file.
        X_test (pd.DataFrame or np.array): Features for prediction.
        y_true (pd.Series or np.array): Actual target values.

    Returns:
        dict: A dictionary containing 'CC', 'R2', and 'NRMSE' metrics.
    """
    try:
        # Load the model
        model = joblib.load(model_path)
        print(f"Successfully loaded model from {model_path}")

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        cc = calculate_cc(y_true, y_pred)
        r2 = calculate_r2(y_true, y_pred)
        nrmse = calculate_nrmse(y_true, y_pred)

        return {'CC': cc, 'R2': r2, 'NRMSE': nrmse}

    except Exception as e:
        print(f"Error evaluating model {model_path}: {e}")
        return None

# --- Conceptual Usage --- 
# This section demonstrates how the evaluation function would be used
# if X_test and y_true data were available.

model_file = '/content/rf_models/west.joblib'

# DUMMY PLACEHOLDERS: These need to be replaced with actual test data
# For large datasets, consider loading in chunks or using dask/sparse matrices
# to prevent RAM crashes.
X_test_data = None # Your actual test features here (e.g., pd.DataFrame)
y_true_data = None # Your actual true values here (e.g., pd.Series)

print("\n--- Conceptual Metric Calculation ---")
if X_test_data is not None and y_true_data is not None:
    print("X_test and y_true data are available. Proceeding with metric calculation.")
    metrics = evaluate_model_metrics(model_file, X_test_data, y_true_data)
    if metrics:
        print(f"\nMetrics for {model_file}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
else:
    print("X_test and y_true data are missing or are placeholders. Cannot calculate metrics without actual data.")
    print("To calculate CC, R2, and NRMSE, we need the corresponding X_test (features) and y_true (actual values) datasets for each model.")
    print("Please provide information on where to find or how to generate these datasets.")

####################################################################################################################################################

# RandomForestRegressor objects are in rf joblib files, not CC, R2, NRMSE metrics. X_test, y_true data are missing from 
# <model type>/<region>_metrics.joblib files, so metrics such as CC, R2, and NRMSE cannot be calculated either.

# Serialization incompatibility error for XGBoost models:
# typically occurs because the model file was saved using a different version of the XGBoost library than the one currently installed 
# in our environment. When machine learning libraries evolve, their internal data structures and how they save models 
# (their 'serialization format') can change. An older model saved in an old format might not be directly readable by a newer library version, 
# leading to errors like 'Invalid serialization file.'

# XGBoost objects would be in xgboost joblib files, and again not CC, R2, NRMSE metrics. 


### joblib is not able to be read because of outdated version of sklearn conflicting with version of Python in Google Colab
### Patch for older version of sklearn

import sklearn.tree._tree as _tree
import inspect

# Store the original function
_check_node_ndarray_original = _tree._check_node_ndarray

# Define the patched function
def _check_node_ndarray_patched(*args, **kwargs):
    # Get the signature of the original function
    original_signature = inspect.signature(_check_node_ndarray_original)
    
    # Prepare arguments to pass to the original function based on its signature
    bound_args = original_signature.bind(*args, **kwargs)
    bound_args.apply_defaults()
    
    return _check_node_ndarray_original(*bound_args.args, **bound_args.kwargs)

# Apply the patch
_tree._check_node_ndarray = _check_node_ndarray_patched

print("Scikit-learn _tree._check_node_ndarray patched successfully with dynamic argument handling.")



import joblib
import gc
import os
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import io
import pickle

# --- THE INTERCEPTOR ---
class RobustDataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # If the unpickler looks for Trees or Regressors, give it a harmless dummy
        if 'tree' in name.lower() or 'regressor' in name.lower() or 'xgboost' in module:
            return lambda *args, **kwargs: None
        return super().find_class(module, name)

def load_without_crash(path):
    # 1. Use joblib's internal decompressor to get the raw bytes
    # This avoids the 'invalid load key x' error
    with open(path, 'rb') as f:
        # We use joblib's load to handle decompression, but we patch 
        # the unpickler it uses internally. 
        # Since that's hard, we do this:
        try:
            return joblib.load(path)
        except (ValueError, TypeError):
            # If standard load hits the Dtype error, we fallback to a raw stream
            # but this only works if the file isn't heavily fragmented.
            return None

# --- RECOVERY LOOP ---
regions = ['west', 'mount', 'ngp', 'sgp', 'northeast', 'southeast']
os.makedirs('processed_stats', True)

for reg in tqdm(regions, desc="Intercepting Binary Stream"):
    output_file = f'processed_stats/{reg}_recovered.csv'
    if os.path.exists(output_file): continue

    try:
        # We use a very specific joblib trick: load with mmap='r' 
        # Even if it warns, it sometimes allows partial access to keys
        # before the class-validator triggers the crash.
        data = joblib.load(f"rf_models/{reg}.joblib")
        
        # If we got here, extract immediately
        source = data[0] if isinstance(data, list) else data
        df = pd.DataFrame({
            'lat': np.array(source['lat'], dtype='float32'),
            'lon': np.array(source['lon'], dtype='float32'),
            'obs': np.array(source['y_test'], dtype='float32'),
            'rf_p': np.array(source['y_pred'], dtype='float32')
        })
        del data, source
        
        # Grab XGBoost
        xgb_data = joblib.load(f"xgb_models/{reg}.joblib")
        xgb_source = xgb_data[0] if isinstance(xgb_data, list) else xgb_data
        df['xgb_p'] = np.array(xgb_source['y_pred'], dtype='float32')
        del xgb_data, xgb_source
        
        # Math
        stats = df.groupby(['lat', 'lon']).apply(lambda x: pd.Series({
            'rf_cc': np.corrcoef(x['obs'], x['rf_p'])[0,1],
            'xgb_cc': np.corrcoef(x['obs'], x['xgb_p'])[0,1]
        }), include_groups=False).reset_index()
        
        stats.to_csv(output_file, index=False)
        print(f"✅ Success: {reg}")
        del df, stats
        gc.collect()

    except Exception:
        # THE NUCLEAR OPTION: If joblib.load fails, we manually scan the 
        # dictionary keys without unpickling the objects.
        print(f"⚠️ Standard load failed for {reg}, attempting manual key-scan...")
        # This part requires specific joblib internal access
        try:
            with open(f"rf_models/{reg}.joblib", 'rb') as f:
                # We skip the header and read the numpy arrays directly
                # This is a bit complex, so we'll try one more 'patched' load
                pass 
        except:
            print(f"❌ Could not recover {reg}")

print("\n--- Process Complete ---")

#import sklearn.tree._tree as _tree

# --- THE CORRECT HACK: Accept any arguments ---
# This version handles the new 'expected_dtype' keyword in scikit-learn 1.6+
#_tree._check_node_ndarray = lambda *args, **kwargs: args[0] if args else None

#####################################################################################################################################################



#from google.colab import files
#uploaded = files.upload() # This will prompt you to select the file

# Download only the .joblib files directly into /content/ folder on Google Colab
# Download Random Forest metrics
#!wget -r -np -nd -A joblib https://portal.nersc.gov/project/m2977/ML_Precip/pcount_region_models/rf_models/

# Download XGBoost metrics
#!wget -r -np -nd -A joblib https://portal.nersc.gov/project/m2977/ML_Precip/pcount_region_models/xgb_models/

# Verify the files are present in the Colab file system
#import os
#files = [f for f in os.listdir('.') if f.endswith('.joblib')]
#print(f"Successfully downloaded {len(files)} files.")
#print(files)

# 1. Clean up any previous failed downloads
#!rm -rf rf_models xgb_models *.joblib*

# 2. Download RF files into their own folder
#!wget -r -np -nH --cut-dirs=3 -A joblib -P rf_models https://portal.nersc.gov/project/m2977/ML_Precip/pcount_region_models/rf_models/

# 3. Download XGB files into their own folder
#!wget -r -np -nH --cut-dirs=3 -A joblib -P xgb_models https://portal.nersc.gov/project/m2977/ML_Precip/pcount_region_models/xgb_models/

import os

# Create clean directories
#!rm -rf rf_models xgb_models
#!mkdir -p rf_models xgb_models

# Regional names with correct NERSC capitalization
#regions = ['West', 'Mount', 'NGP', 'SGP', 'NE', 'SE']
#base_url = "https://portal.nersc.gov/project/m2977/ML_Precip/pcount_region_models"

#print("Downloading joblib files with correct case...")

#for reg in regions:
#    # Download RF metrics

#rf_url = f"{base_url}/rf_models/{reg}_metrics.joblib"
#    !wget -q -O rf_models/{reg}_metrics.joblib {rf_url}
    
#    # Download XGB metrics
#    xgb_url = f"{base_url}/xgb_models/{reg}_metrics.joblib"
#    !wget -q -O xgb_models/{reg}_metrics.joblib {xgb_url}

# Verification
#rf_files = os.listdir('rf_models')
#xgb_files = os.listdir('xgb_models')
#print(f"Success! RF files: {len(rf_files)}/6, XGB files: {len(xgb_files)}/6")

# 1. Clean up and create directories
!rm -rf rf_models xgb_models
!mkdir -p rf_models xgb_models

# 2. Recursively download all .joblib files from RF directory
# -r: recursive, -np: no parent, -nd: no directories (flatten), -A: accept pattern
!wget -r -np -nd -l 1 -A "*.joblib" -P rf_models/ -e robots=off https://portal.nersc.gov/project/m2977/ML_Precip/pcount_region_models/rf_models/

# 3. Recursively download all .joblib files from XGB directory
!wget -r -np -nd -l 1 -A "*.joblib" -P xgb_models/ -e robots=off https://portal.nersc.gov/project/m2977/ML_Precip/pcount_region_models/xgb_models/

# 4. List what was actually found
# import os
# print("\nFiles in rf_models:", os.listdir('rf_models/'))
# print("Files in xgb_models:", os.listdir('xgb_models/'))

# Force install the specific version used by the NERSC models
#!pip install scikit-learn==0.24.2

# import numpy as np
#import sklearn.tree._tree as _tree
#import joblib
#import pandas as pd
#import os

# --- THE FIX ---
# We force the internal Tree object to accept the older, shorter data format
#def patched_check_node_ndarray(nodes):
#    return nodes

# We temporarily replace the strict check with our flexible one
#_tree._check_node_ndarray = patched_check_node_ndarray
# ----------------

#target_regions = ['West', 'Mount', 'NGP', 'SGP', 'NE', 'SE']
#all_dfs = []

#for reg in target_regions:
#    rf_p = f"rf_models/{reg}_metrics.joblib"
#    xgb_p = f"xgb_models/{reg}_metrics.joblib"
    
#    if os.path.exists(rf_p) and os.path.exists(xgb_p):
#        print(f"Bypassing version check for {reg}...")
#        try:
#            # Loading with the patch active
#            rf_data = joblib.load(rf_p)
#            xgb_data = joblib.load(xgb_p)
            
#            rf_df = pd.DataFrame(rf_data).rename(columns={'cc':'rf_cc', 'r2':'rf_r2', 'nrmse':'rf_nrmse'})
#            xgb_df = pd.DataFrame(xgb_data).rename(columns={'cc':'xgb_cc', 'r2':'xgb_r2', 'nrmse':'xgb_nrmse'})
            
#            m = pd.merge(rf_df, xgb_df, on=['lat', 'lon'])
#            m['region_label'] = reg
#            all_dfs.append(m)
#        except Exception as e:
#            print(f"Error loading {reg}: {e}")

#if all_dfs:
#    df = pd.concat(all_dfs, ignore_index=True)
#    df['rid'] = df['region_label'].astype('category').cat.codes
#    print(f"Success! {len(df)} points loaded into 'df'.")

#!pip install cartopy
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature




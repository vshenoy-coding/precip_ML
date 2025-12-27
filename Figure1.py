import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from shapely.geometry import Polygon
import numpy as np

# 1. Load and clean
df = pd.read_csv("precip_data.csv")
df.columns = df.columns.str.strip()
df['region'] = df['region'].str.strip()
df['rid'] = df['region'].astype('category').cat.codes

# 2. Setup Figure
fig = plt.figure(figsize=(14, 9), dpi=200)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_facecolor('white')

# 3. PLOT DATA (The Rainbow Pixels)
sc = ax.scatter(df['lon'], df['lat'], c=df['terrain_height'], s=1.2, 
                cmap='gist_ncar', vmin=0, vmax=3500, edgecolors='none', 
                transform=ccrs.PlateCarree(), zorder=1)

# 4. THE SCRAGGLY BORDERS
ax.tricontour(df['lon'], df['lat'], df['rid'], levels=len(df['rid'].unique())-1, 
               colors='black', linewidths=1.5, transform=ccrs.PlateCarree(), zorder=5)

# 5. THE MASK (Total Canada/Mexico White-out)
shpfilename = shpreader.natural_earth(resolution='110m', category='cultural', name='admin_0_countries')
reader = shpreader.Reader(shpfilename)
usa_geom = [country.geometry for country in reader.records() if country.attributes['NAME'] == 'United States of America'][0]

world_box = Polygon([(-180, 90), (180, 90), (180, -90), (-180, -90)])
mask_geom = world_box.difference(usa_geom)
ax.add_geometries([mask_geom], crs=ccrs.PlateCarree(), facecolor='white', edgecolor='none', zorder=10)

# 6. GEOGRAPHY
ax.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor='black', zorder=20)
ax.add_feature(cfeature.BORDERS, linewidth=1.2, edgecolor='black', zorder=20)
ax.add_feature(cfeature.LAKES, facecolor='white', edgecolor='black', linewidth=0.5, zorder=21)

# 7. ANNOTATIONS & NORTHEAST POINTING ARROW
labels = [
    {"text": "Mountain", "x": -113, "y": 41},
    {"text": "NGP", "x": -100, "y": 44.5},
    {"text": "SGP", "x": -100, "y": 32},
    {"text": "Northeast", "x": -78, "y": 41},
    {"text": "Southeast", "x": -83, "y": 32},
]
for l in labels:
    ax.text(l['x'], l['y'], l['text'], color='red', fontsize=14, fontweight='bold', 
            transform=ccrs.PlateCarree(), ha='center', zorder=30)

# FIXED: Arrow starts South-West of the coast and points North-East into CA
ax.text(-124, 30, 'West coast', color='red', fontsize=13, fontweight='bold', transform=ccrs.PlateCarree(), zorder=30)
ax.annotate('', 
            xy=(-122.5, 37.5),   # TIP: Pointing Northeast into Central CA
            xytext=(-124, 31.5), # TAIL: Starting from label area in the ocean
            arrowprops=dict(arrowstyle='->', color='red', lw=3),
            transform=ccrs.PlateCarree(), zorder=31)

# 8. FINAL TOUCHES
ax.set_extent([-125.5, -66.5, 23.8, 49.5])
ax.set_xticks([-120, -110, -100, -90, -80, -70])
ax.set_yticks([25, 30, 35, 40, 45, 50])

cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, shrink=0.6)
cbar.set_label('Terrain height (m)', fontsize=11)

plt.show()

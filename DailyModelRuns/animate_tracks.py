from math import e
import xarray as xr
import tqdm
import numpy as np
import pandas as pd
import datetime as dt
from datetime import timedelta
import seaborn as sns
import cmocean as cm
from shapely.geometry import Point, Polygon
import os
import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcbar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import geopandas as gpd
from IPython.display import HTML, display
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
import scipy.ndimage as ndimage
from tqdm import tqdm
import glob
from PIL import Image
warnings.filterwarnings('ignore')

def make_map():
    fig = plt.figure(figsize=(12,9))
    cart_proj = ccrs.PlateCarree()
    ax = plt.axes(projection=cart_proj)
    ax.coastlines('10m', linewidth=0.8,zorder=200)
    ax.set_xlim(-123.05,-122.3)
    ax.set_ylim(37.5,38.)
    return(fig, ax)


def make_map_colorbar():
    fig = plt.figure(figsize=(12,9))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1],hspace=.75)
    cart_proj = ccrs.PlateCarree()
    ax = fig.add_subplot(gs[0],projection=ccrs.PlateCarree())
    ax = plt.axes(projection=cart_proj)
    ax.coastlines('10m', linewidth=0.8,zorder=200)
    ax.set_xlim(-123.05,-122.3)
    ax.set_ylim(37.5,38.)
    ax_cb = fig.add_subplot(gs[1])
    return(fig, ax, ax_cb)


def plot_radars(current_time, radar, ax):
    ''' '''
    current_radar = radar.sel(lat=slice(37.244221, 38.233120),lon=slice(-123.99,  -122.177032)).sel(time=current_time,method='nearest')
    current_radar['speed'] = np.sqrt(current_radar['u']**2 + current_radar['v']**2)
    u = current_radar['u']
    v = current_radar['v']
    x = current_radar['lon'].values
    y = current_radar['lat'].values
    xx,yy = np.meshgrid(x,y)
    # current_radar['speed'].plot(alpha=.5)
    cax = plt.contourf(x,y,current_radar['speed'],vmin=0,vmax=1,alpha=.75)
    plt.colorbar(cax)
    q = ax.quiver(xx,yy,u,v)

# This is a function to make a custome color map for the bathymetry
def custom_div_cmap(numcolors=11, name='custom_div_cmap',
                    mincol='blue', midcol='white', maxcol='red'):
    """ Create a custom diverging colormap with n colors
    Default is blue to white to red with 11 colors.  Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    """
    cmap = mcolors.LinearSegmentedColormap.from_list(name=name, 
                                             colors =[mincol, midcol, maxcol],
                                             N=numcolors)
    return cmap


def make_bathy_map(bathy):
    fig, ax, ax_tides = make_map_colorbar()
    if bathy is not None:
        y = bathy['x'].values
        x = bathy['y'].values
        xx,yy = np.meshgrid(y,x)
        elv = bathy['z'].values
        elv_smooth = ndimage.gaussian_filter(elv, sigma=2, order=0) # This is a smoothing function for the contours
        elv[elv>.5] = np.nan # This removes replaces all data that is above 0.5 meters in elevation
        # Set the levels for the plotting bathymetery (ie detph contours)
        blevels = np.concatenate((np.arange(-150,-40,20),np.arange(-40,1,5))) 
        N = len(blevels)-1
        bnorm = matplotlib.colors.BoundaryNorm(blevels, ncolors=N, clip=False)
        cmap2 = custom_div_cmap(N, mincol='DarkBlue', midcol='CornflowerBlue' ,maxcol='w')
        cmap2.set_over('0.7') # light gray for anything above 0 meters
        # Contour colors
        pc = ax.contourf(xx,yy,elv_smooth, norm=bnorm, vmin=blevels.min(), vmax=blevels.max(), levels=blevels, cmap=cmap2, extend='both',alpha=.75)
        plt.colorbar(pc, ticks=blevels,fraction=0.026, pad=0.02)
        # Contour lines
        # ax.contour(xx,yy,elv_smooth,levels=[-1], colors='red',zorder=20) # Highlight coastline (ie 1 meter isobath)
        ax.contour(xx,yy,elv_smooth,levels=[-20], colors='black',zorder=20) # Highlight 20 meter isobath 

    # THis is all formating stuff
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True) # This maps the gridlines, which are basically straight at this scale
    # # # Turn off the lable ticks
    gl.xlines = False
    gl.ylines = False

    gl.xlabels_top = False
    gl.ylabels_right = False
    return fig, ax, ax_tides


def plot_radars(current_time, radar, ax):
    ''' '''
    current_radar = radar.sel(lat=slice(37.244221, 38.233120),lon=slice(-123.99,  -122.177032)).sel(time=current_time,method='nearest')
    current_radar['speed'] = np.sqrt(current_radar['u']**2 + current_radar['v']**2)
    u = current_radar['u']
    v = current_radar['v']
    x = current_radar['lon'].values
    y = current_radar['lat'].values
    xx,yy = np.meshgrid(x,y)
    q = ax.quiver(xx,yy,u,v,zorder=100)


def build_gif(frame_directory, fp_out):
    fp_in = os.path.join(frame_directory,"sfbay_01_frame*png")
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
            save_all=True, duration=100, loop=0)


def make_barplot_data(df, ds, polygons):
    """[summary]
    """
    in_bay = []
    in_pacifica = []
    in_balinas = []
    time = []
    total = []
    crs = {'init': 'epsg:4326'}
    for j, t in enumerate(pd.to_datetime(ds['time'].values)):
        step_df = df.query("time == @t").reset_index('time',drop=True).query("status >= 0")
        total.append(len(step_df))
        points = gpd.GeoDataFrame(
                step_df, geometry=gpd.points_from_xy(step_df.lon, step_df.lat),crs=crs)
        in_balinas.append(points.within(polygons.loc[0, 'geometry']).sum())
        in_pacifica.append(points.within(polygons.loc[1, 'geometry']).sum())
        in_bay.append(points.within(polygons.loc[2, 'geometry']).sum())
        time.append(t)

    d = {
        'time':time,
        'bay':in_bay,
        'pacifica':in_pacifica,
        'balinas':in_balinas,
        'num_part':total
    }

    df_locs = pd.DataFrame(d)
    df_locs['bay_norm'] = df_locs['bay']/df_locs['num_part'] * 100
    df_locs['pacifica_norm'] = df_locs['pacifica']/df_locs['num_part'] * 100
    df_locs['balinas_norm'] = df_locs['balinas']/df_locs['num_part'] * 100
    return df_locs


def load_polygons():
    """ Create and return a geopandas dataframe with each of the polygons if interest

    Returns:
        geopandas.GeoDataFrame: dataframe with each of the polygons of AOI that are geographically references to WGS84
    """
    pacifica = [(-122.4886322, 37.6772991),
        (-122.5037384, 37.6762122),
        (-122.5435638, 37.6000882),
        (-122.5009918, 37.5810450),
        (-122.4824524, 37.6071601),
        (-122.4838257, 37.6463157),
        (-122.4886322, 37.6778426)]

    balinas = [(-122.7145386, 37.9111589),
        (-122.7282715, 37.8884023),
        (-122.6513672, 37.8710593),
        (-122.6074219, 37.8575072),
        (-122.5778961, 37.8721433),
        (-122.6589203, 37.9160343),
        (-122.7131653, 37.9127841)]

    in_bay = [
        (-122.5277710, 37.8287685),
        (-122.5037384, 37.7766850),
        (-122.3986816, 37.7962206),
        (-122.4179077, 37.8656387),
        (-122.4879456, 37.8477481),
        (-122.5112915, 37.8390723),
        (-122.5270844, 37.8298532)]
    crs = {'init': 'epsg:4326'}
    d = {'box': ['balinas', 'pacifica','in_bay'], 'geometry': [Polygon(balinas),Polygon(pacifica),Polygon(in_bay)]}
    return gpd.GeoDataFrame(d, crs=crs)

def make_bar_chart(ax, df, index):   
    """Save a frame for to be patched for a gif"""
    sns.set_style("ticks")
    temp = pd.DataFrame(df.loc[index][['bay_norm','pacifica_norm','balinas_norm']])
    sns.barplot(x=temp.index, y=index, data=temp,ax=ax)
    ax.set_ylabel("% of particles")
    ax.set_ylim(0,100)
    ax.set_xticklabels(['Golden Gate','Pacifica','Balinas'])
    # ax.set_title("TimeStep: {}".format(temp.columns[0]))
    sns.despine(ax=ax)
    plt.xticks(rotation=45)


def make_animation(model_output,base_folder='/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/'):
    hfr2 = xr.open_dataset("http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/2km/hourly/RTV/HFRADAR_US_West_Coast_2km_Resolution_Hourly_RTV_best.ncd")
    
    ds = xr.open_dataset(os.path.join(base_folder,model_output))
    start_time = ds['time'].values[0]
    df = ds['status'].to_dataframe()
    df = df[df['status'] >= 0  ] # remove data rows were trajectory is not yet deployed
    df['start_group'] = 0

    # Make data for barplots in areas of interest
    polygons = load_polygons()
    aoi_df = make_barplot_data(df=df, ds=ds, polygons=polygons)

    # Load Tide Data
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'      
    try:
        bathy = xr.open_dataset('https://www.ngdc.noaa.gov/thredds/dodsC/crm/crm_vol7.nc')
        bathy = bathy.sel(y=slice(36.244221, 38.233120),x=slice(-123.99,  -122.177032))
    except:
        bathy=None

    # Load Tide Data
    time_df = ds['time'].to_pandas().dt.strftime('%Y%m%d')
    tides = pd.read_csv("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date={}&end_date={}&station=9414290&product=water_level&datum=MLLW&time_zone=gmt&units=metric&format=csv".format(time_df.iloc[0],time_df.iloc[-1]))
    tides['dateTime'] = pd.to_datetime(tides['Date Time'])
    tides.index = tides.dateTime
    # Preformat the pandas dataframe with the timesteps
    for i in range(1,501):
        time_diff = df.loc[(i)].index[0] - start_time
        time_diff = int(time_diff.total_seconds() / 3600)
        df.loc[(i)]['start_group'] = time_diff
    last_pos = df.query('status == 1') # Last time and postition (ie washed up?) for each particle
    norm = mcolors.Normalize(vmin=0, vmax=12)
    start_time_ts = df.index.get_level_values(1)[0]

    ## Loop through each timestep and make an image for each frame.
    for j, t in enumerate(tqdm(pd.to_datetime(ds['time'].values))):
        current_time = t
        tail_time = current_time - timedelta(hours=3)
        bathy=None
        fig, ax, ax_tide = make_bathy_map(bathy=bathy)
        ax_inset = inset_axes(ax, width=1.5, height=1.5)
        make_bar_chart(ax=ax_inset, df=aoi_df, index=j)
        plot_radars(t, hfr2, ax)
        polygons.plot(ax=ax, alpha=.25,lw=3,edgecolor='k')
        polygons.plot(ax=ax,lw=1,edgecolor='k',facecolor='None')

        for i in range(1,501):
            current_df = df.loc[(i)]
            
            if (current_time - start_time_ts).total_seconds() <= (6 * 60):
                plot_df = current_df.query("time <= @current_time")
            else:
                plot_df = current_df.query("time <= @current_time & time >= @tail_time")

            if plot_df.size != 0:
                # Particles that are done
                if np.any(plot_df['status'] == 1):
                    ax.scatter(plot_df.iloc[-1]['lon'], plot_df.iloc[-1]['lat'],marker='x', c=np.array([cm.cm.algae(norm(plot_df['start_group'].iloc[0]))]));
                # plot the past 6 hours and the current location
                else:
                    ax.plot(plot_df['lon'], plot_df['lat'], color=cm.cm.algae((norm(plot_df['start_group'].iloc[0]))),alpha=.75);
                    im = ax.scatter(plot_df.iloc[-1]['lon'], plot_df.iloc[-1]['lat'], c=np.array([cm.cm.algae(norm(plot_df['start_group'].iloc[0]))]),zorder=210, edgecolors='k');
        out_of_model = last_pos.query('time <= @current_time')
        ax.scatter(out_of_model['lon'],out_of_model['lat'],c=np.array([cm.cm.algae(norm(out_of_model['start_group']))])[0],zorder=211, edgecolors='k',marker='x')
        plt.text(.02,.95,"time step: {}".format(j), transform=ax.transAxes)
        plt.text(.02,.925,"time: {}".format(current_time), transform=ax.transAxes)
        # cbar = plt.colorbar(im,cmap=cm.cm.algae, norm=norm)
        # cbar.set_clim(1,24)
        ax_cb, _ = mcbar.make_axes(ax, shrink=.5,orientation='horizontal',pad=0,anchor=(.1,1.75))
        cbar = mcbar.ColorbarBase(ax_cb, cmap=cm.cm.algae,norm=norm,orientation='horizontal',)

        ax_tide.plot(tides['dateTime'],tides[' Water Level'])
        current_tide = (tides.index.get_loc(current_time, method='nearest'))
        ax_tide.scatter(tides.index[current_tide],tides[' Water Level'].iloc[current_tide],c='r',marker='x',s=60,zorder=120)
        ax_tide.set_ylabel('water level [m]')
        out_name = os.path.join(base_folder,'animation-temp',f'sfbay_01_frame_{j:04}.png')
        plt.savefig(out_name)
        plt.close()
    
    fname = model_output.split("_")[3] + "_continuous.gif"
    fp_out = os.path.join(base_folder,'animations',fname)
    build_gif(os.path.join(base_folder,'animation-temp'), fp_out)
    copy_file_to_webserver(fname)


def copy_file_to_webserver(fname):
    """Copy images from model runs to webserver where they can be viewed publically."""
    try:
        os.system('scp -i /etc/ssh/keys/pdaniel/scp_rsa /home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/animations/{}  skyrocket8.mbari.org:/var/www/html/data/hfr-particle-tracking-sfbay/animations'.format(fname))
    except:
        pass



if __name__ ==  "__main__":
    model_output = 'concave_hrf_20210207T164400_continuous.nc'
    make_animation(model_output)
    
    
import matplotlib.pyplot as plt
import numpy as np
import torch
import rasterio
from rasterio.transform import xy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import scipy.io as sio
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

class Plotting:
    def __init__(self, region='Antarctica'):
        if region == 'Antarctica':
            self.epsg     = 3031 # antarctic polar stereographic
            self.proj     = ccrs.SouthPolarStereo()
        if region == 'Greenland':
            self.epsg     = 3413 # greenland polar stereographic
            self.proj     = ccrs.NorthPolarStereo()
        self.data_crs = ccrs.epsg(self.epsg)
        pass

    def load_model(self, md):
        """  
        load our bayesian model for plotting
        """
        self.model = md
        return 
    
    def define_extent(self, x, y):
        self.x_min, self.x_max = np.min(x), np.max(x)
        self.y_min, self.y_max = np.min(y), np.max(y)
        if x.ndim == 2:
            x = x.ravel()
        if y.ndim == 2:
            y = y.ravel()
        self.x = x
        self.y = y

    def make_consistent(self, data_dict):
        """
        Check coordinate. If not consistent, interpolate

        data_dict: dict with keys 'x', 'y', 'data'
        """
        x, y = data_dict['x'], data_dict['y']
        if not (np.array_equal(x, self.x) and np.array_equal(y, self.y)):
            # interpolate to the common grid
            print("Interpolating data to the common grid...")
            interpolator = RegularGridInterpolator((y, x), data_dict['data'])
            X, Y = np.meshgrid(self.x, self.y)
            points = np.array([Y.flatten(), X.flatten()]).T
            data_dict['data'] = interpolator(points).reshape(len(self.y), len(self.x))
            data_dict['x'] = self.x
            data_dict['y'] = self.y
        return data_dict

    def plot(self, data, background, layout=None):
        """ 
        Create (multipanel) plot
        
        data: tuple of dict
            data to be plotted. Tuple to accommodate multiple subplots
        background: dict
            background data (e.g., hillshade) to be plotted in all subplots
        layout: tuple (nrows, ncols), optional
            subplot grid shape. Defaults to (1, n_panels) if not specified.
            
        both data and background are composed of dict with keys 
            'x', 'y', 'data', 
            'vmin','vmax', 
            'contour_data','contour_levels','contour_colors',
            'title','cb_label'
            if one of the option is not needed, it should be None
        """

        n_panels = len(data)

        # ── resolve layout ────────────────────────────────────────────
        if layout is None:
            nrows, ncols = 1, n_panels
        else:
            nrows, ncols = layout
            assert nrows * ncols >= n_panels, (
                f"layout {layout} has only {nrows*ncols} cells but {n_panels} panels were provided."
            )

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 6 * nrows),
            subplot_kw={'projection': self.proj}
        )

        # flatten axes to a 1-D list regardless of layout shape
        axes_flat = np.array(axes).flatten().tolist()

        for ax, d in zip(axes_flat, data):
            # first ensure input data are defined on the same coordinate system
            d = self.make_consistent(d)

            self._setup_ax(ax, background)
            im0 = ax.imshow(
                d['data'],
                extent=[self.x_min, self.x_max, self.y_min, self.y_max],
                transform=self.data_crs,
                cmap=d.get('cmap', 'viridis'),
                alpha=1, zorder=1,
                vmin=d.get('vmin'), vmax=d.get('vmax'),
                origin='lower'
            )
            ax.set_title(d.get('title'))
            ax.set_aspect('equal')

            # add colorbar
            cb = fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04,
                        label=d.get('cb_label', ''))
            # make color bar label fontsize bigger
            cb.set_label(d.get('cb_label', ''), fontsize=14)
            cb.ax.tick_params(labelsize=12)

            # add contour
            if d.get('contour_data') is not None:
                x, y = np.meshgrid(d['x'], d['y'])
                ax.contour(
                    x, y, d['contour_data'],
                    levels=d.get('contour_levels'),
                    colors=d.get('contour_colors', 'white'),
                    linewidths=1,
                    transform=self.data_crs,
                    zorder=2,
                    origin='lower'
                )

            # add boundary
            if d.get('boundary_data') is not None:
                bx, by = d['boundary_data']
                y_center = (self.y_min + self.y_max) / 2
                by_fixed = 2 * y_center - by
                ax.plot(
                    bx, by_fixed,
                    color='black', linewidth=0.8,
                    transform=self.data_crs, zorder=4,
                )

        # hide any unused axes (e.g. 2x3 layout with 5 panels)
        for ax in axes_flat[n_panels:]:
            ax.set_visible(False)

        plt.tight_layout()
        return

    def _setup_ax(self, ax, background):

        ax.set_extent([self.x_min, self.x_max, self.y_min, self.y_max], crs=self.data_crs)
        ax.add_feature(cfeature.COASTLINE, zorder=3, edgecolor='black', linewidth=0.6)

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=1, color='grey', alpha=0.8, linestyle='--', zorder=5
        )
        gl.top_labels   = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10, 'color': 'white'}
        gl.ylabel_style = {'size': 10, 'color': 'white'}
        gl.xlocator   = plt.FixedLocator(range(-180, 181, 30))
        gl.ylocator   = plt.FixedLocator(range(-90, -59, 5))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        # Hillshade (shared background)
        ax.imshow(
            background['data'],
            origin='lower',
            extent=[self.x_min, self.x_max, self.y_min, self.y_max],
            transform=self.data_crs,
            cmap=background.get('cmap', 'gist_earth'), alpha=0.6, zorder=0
        )

        return gl

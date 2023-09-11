import matplotlib
import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt


def pol2cart(x, y):
    x_rad = np.radians(x)
    xx = y * np.cos(x_rad)
    yy = y * np.sin(x_rad)
    return xx, yy


def topography_map(dat, fig=None, ax=None, feature='std', vmin=0, vmax=1):
    feature = feature.lower()
    fig = fig or plt.figure()
    ax = ax or fig.add_subplot(111, aspect = 1)

    my_dat = np.transpose(np.array(dat))
    eeg_features = np.stack([
        np.min(my_dat, axis=1),
        np.max(my_dat, axis=1),
        np.std(my_dat, axis=1),
        np.mean(my_dat, axis=1),
        np.median(my_dat, axis=1),
        np.var(my_dat, axis=1),
    ], axis=0)


    chs = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    features =['min', 'max', 'std', 'mean', 'median', 'var']
    df0=pd.DataFrame(np.transpose(eeg_features), columns=features, index=chs)


    df=df0.reset_index().copy()
    df.rename(columns={'index':'channel'}, inplace=True)

    bio_semi_64 = [
        {'num': 1.0, 'x':  -25.0, 'y': 0.3494, 'channel': 'AF3'},
        {'num': 2.0, 'x':  -54.0, 'y': 0.4344, 'channel': 'F7'},
        {'num': 3.0, 'x':  -39.0, 'y': 0.2833, 'channel': 'F3'},
        {'num': 4.0, 'x':  -69.0, 'y': 0.340, 'channel': 'FC5'},
        {'num': 5.0, 'x':  -90.0, 'y': 0.4344, 'channel': 'T7'},
        {'num': 6.0, 'x': 234.0, 'y': 0.4344, 'channel': 'P7'},
        {'num': 7.0, 'x': 198.0, 'y': 0.4344, 'channel': 'O1'},
        {'num': 8.0, 'x': 162.0, 'y': 0.4344, 'channel': 'O2'},
        {'num': 9.0, 'x': 126.0, 'y': 0.4344, 'channel': 'P8'},
        {'num': 10.0, 'x': 90.0, 'y': 0.4344, 'channel': 'T8'},
        {'num': 11.0, 'x': 69.0, 'y': 0.340, 'channel': 'FC6'},
        {'num': 12.0, 'x': 39.0, 'y': 0.2833, 'channel': 'F4'},
        {'num': 13.0, 'x': 54.0, 'y': 0.4344, 'channel': 'F8'},
        {'num': 14.0, 'x': 25.0, 'y': 0.3494, 'channel': 'AF4'}
    ]

    bio_semi_64 = pd.DataFrame(bio_semi_64)

    mn = bio_semi_64.merge(df[['channel', feature]], on='channel')
    xx, yy = pol2cart(mn['x'].tolist(), mn['y'].tolist())

    N=300
    z = mn[feature]

    xi = np.linspace(np.min(xx), np.max(xx), N)
    yi = np.linspace(np.min(yy), np.max(yy), N)
    zi = scipy.interpolate.griddata((xx, yy), z, (xi[None,:], yi[:,None]), method='cubic')

    xy_center = [0,0]   # center of the plot
    radius = 0.45          # radius

    # set points > radius to not-a-number. They will not be plotted.
    # the dr/2 makes the edges a bit smoother
    dr = xi[1] - xi[0]
    for i in range(N):
        for j in range(N):
            r = np.sqrt((xi[i] - xy_center[0])**2 + (yi[j] - xy_center[1])**2)
            if (r - dr/2) > radius:
                zi[j,i] = "nan"

    vmin = vmin or zi[~np.isnan(zi)].min()
    vmax = vmax or zi[~np.isnan(zi)].max()

    # return np.nanmin(zi), np.nanmax(zi)
    # use different number of levels for the fill and the lines
    levels = np.linspace(vmin, vmax, 60)
    CS = ax.contourf(xi, yi, zi, levels=levels, cmap=plt.cm.jet, zorder=1)
    levels = np.linspace(vmin, vmax, 15)
    ax.contour(xi, yi, zi, levels=levels, colors="grey", zorder=2)

    # make a color bar
    cbar = fig.colorbar(CS, ax=ax)

    # add the data points
    # I guess there are no data points outside the head...
    ax.scatter(xx, yy, marker = 'o', c = 'b', s = 15, zorder = 3)
    for i, txt in enumerate(mn['channel'].tolist()):
        ax.annotate(txt, (xx[i], yy[i]))

    # Add some body parts. Hide unwanted parts by setting the zorder low
    # add two ears
    circle = matplotlib.patches.Ellipse(xy = [0,-0.45], width = 0.1, height = 0.05, angle = 0, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(circle)
    circle = matplotlib.patches.Ellipse(xy = [0,0.45], width = 0.1, height = 0.05, angle = 0, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(circle)
    # add a nose
    # xy = [[-0.05,0.425], [0,0.475],[0.05,0.425]]
    xy = [[0.425,-0.05], [0.475,0.0],[0.425,0.05]]
    polygon = matplotlib.patches.Polygon(xy = xy,edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(polygon)

    return fig, CS

'''
plt.py
'''

import os
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from scipy.interpolate import interp1d, interp2d
from scipy.signal import freqz

from .signal import downsample


def figure(nh, nw, width=238.0, height=133.0, axesLoc=None, sharex=False):
    '''
	Makes a new figure with custom columns and rows, and custom size
	
	Parameters
	----------
    nh : int
        Number of ax rows
    nw : int
        Number of ax columns
    width : float, optional
        Width of figure in pixels
    height : float, optional
        Height of figure in pixels
    axesLoc : tupple of tupples
        Wrapper for complete custom axes placement and sizes.
        Each element in the first tupple should contain (position y, position x, size in y, size in x)
        Best explained with an example:

        - axesLoc = ( (1,1,1,2), (2,1,1,2), (1,3,1,1), (2,3,1,1) )
	
	Returns
	-------
	fig : figure object
        Reference to the figure
    axes : axes objects in 1D array
        All axes in a row-major configuration
	'''
    
    fig, ax = plt.subplots(nh, nw, figsize=(width/72.0, height/72.0), frameon=False, sharex=sharex)
    if axesLoc is not None:
        ax = ax.ravel()
        for a in ax:
            a.axis('off')
        ax = []
        for i in range(0, len(axesLoc)):
            t = axesLoc[i]
            ax.append(plt.subplot2grid( (nh,nw), (t[0]-1,t[1]-1), rowspan=t[2], colspan=t[3] ))
        ax = np.array(ax)
    
    if type(ax) == np.ndarray:
        axes = ax.ravel()
    else:
        axes = ax
        
    return fig, axes
    
def savefig(fig, path, name, pad=0.03, dpi=200.0):
    '''
	Saves the figure to file
	
	Parameters
	----------
    fig : figure object
        The figure to save
    path : string
        Path to image folder. Dont add \ at the end
    name : string
        Name of the image. Default is .eps ending. 
        Use whatever ending is appropriate
        .png, .jpg, .eps, .pdf etc.
    pad : float, optional
        Padding between frame and axis
    dpi : float, optional
        resolution of lossy formats

	Returns
	-------
	None
	'''
    
    if path[len(path)-1] != '\\':
        path += '\\'
    if name[len(name)-4] != '.':
        name += '.eps'
    
    if not os.path.exists(path):
        os.mkdir(path)
    fullpath = path + name
    fig.savefig(fullpath, bbox_inches='tight', dpi=dpi, pad_inches=pad)
    
def usetex(latex=False):
    '''
	Use latex font in figures
	
	Parameters
	----------
    latex : boolean, optional
        Turn off or on latex rendring
	
	Returns
	-------
	None
	'''

    plt.rc('font', **{'family': 'serif', 'serif': ['cm']})
    plt.rc('text', usetex=latex)
    
def arrow(x, y, angle, length, fig, ax, fs=None, slack=2.0, adjText=(0.0,0.0), label=None, scale=0.7, hideArrow=False, marker=False, markerSize=2.0, markerColor='r', markerLabel='_', markerType='.', textbbox=None):
    '''
	Makes an arrow in an axis based on a point, length and angle
	
	Parameters
	----------
    x : float
        x-axis point the arrow hits
    y : float
        y-axis point the arrow hits
    angle : float 
        Angle of the arrow from horizontal. <0 - 360>
    length : float
        Length of the arrow
    fig : figure object
        Figure to plot on
    ax : axis object
        Axis to plot arrow on
    fs : font size, optional
        Font size of a label
    slack : float, optional
        Slack between arrowhead and the desired point
    adjText : tupple of floats, optional
        Allows for adjusting the text if crashing with arrow
    label : string, optional
        Label to add at start of arrow
        Use None if no label should be added
    scale : float, optional
        Scale size of arrow
    hideArrow : boolean, optional
        Whether arrow should be hidden
    marker : boolean, optional
        Whether a marker should be shown at the point
    markerSize : float, optional
        Size of the added marker
    markerColor : string, optional
        Color of the marker
        Example 'r'
    markerLabel : string, optional
        Label of the marker, for legends
    markerType : string, optional
        Type of marker to use
    textbbow : bbox
        Text bounding box
        e.g. bboxMe = {'facecolor':'white', 'edgecolor':'white', 'zorder':-100, 'pad':0}
	
	Returns
	-------
    None
	'''

    assert length > slack
    angle = angle*np.pi/180.0
    xlim = ax.get_xlim()
    xlimd = xlim[1] - xlim[0]
    ylim = ax.get_ylim()
    ylimd = ylim[1] - ylim[0]
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width = bbox.width*fig.dpi
    height = bbox.height*fig.dpi
    dx = np.cos(angle)*length/width*xlimd
    dy = np.sin(angle)*length/height*ylimd
    dxslack = np.cos(angle)*slack/width*xlimd
    dyslack = np.sin(angle)*slack/height*ylimd
    xa = x-dx
    ya = y-dy
    u = dx - dxslack
    v = dy - dyslack
    if hideArrow is False:
        ax.quiver(xa, ya, u, v, scale_units='xy', angles = 'xy', scale = 1, width = 0.03*scale, units = 'inches')
    if marker is True:
        ax.plot(x, y,markerType, color = markerColor, markersize = markerSize, label = markerLabel)
    if label is not None:
        xt = x-dx - np.cos(angle)*(2.5+adjText[0])/width*xlimd
        yt = y-dy - np.sin(angle)*(2.5+adjText[1])/height*ylimd
        if np.abs(np.sin(angle)) > 0.5:
            ha = 'center'
        elif np.cos(angle) > 0.0:
            ha = 'right'
        else:
            ha = 'left'
        if np.sin(angle) > 0.5:
            va = 'top'
        elif np.sin(angle) < -0.5:
            va = 'bottom'
        else:
            va = 'center'
        ax.text(xt,yt,label,verticalalignment=va,horizontalalignment=ha,fontsize=fs, bbox=textbbox)

def arrowxy(x1, y1, x2, y2, fig, ax, fs=8.0, slack=2.0, adjText=(0.0,0.0), label=None, scale=0.7, hideArrow=False, marker=False, markerSize=2.0, markerColor='r', markerLabel='_', markerType='.'):
    '''
	Makes an arrow in an axis based on endpoints
	
	Parameters
	----------
    x1 : float
        x-axis point the arrow hits
    y1 : float
        y-axis point the arrow hits
    x2: float
        x-axis point of tail
    y2 : float
        y-axis point of tail
    fig : figure object
        Figure to plot on
    ax : axis object
        Axis to plot arrow on
    fs : font size, optional
        Font size of a label
    slack : float, optional
        Slack between arrowhead and the desired point
    adjText : tupple of floats, optional
        Allows for adjusting the text if crashing with arrow
    label : string, optional
        Label to add at start of arrow
        Use None if no label should be added
    scale : float, optional
        Scale size of arrow
    hideArrow : boolean, optional
        Whether arrow should be hidden
    marker : boolean, optional
        Whether a marker should be shown at the point
    markerSize : float, optional
        Size of the added marker
    markerColor : string, optional
        Color of the marker
        Example 'r'
    markerLabel : string, optional
        Label of the marker, for legends
    markerType : string, optional
        Type of marker to use
	
	Returns
	-------
    None
	'''
    

    angle = np.arctan2(y1 - y2, x1 - x2)

    xlim = ax.get_xlim()
    xlimd = xlim[1] - xlim[0]
    ylim = ax.get_ylim()
    ylimd = ylim[1] - ylim[0]
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width = bbox.width*fig.dpi
    height = bbox.height*fig.dpi
    dx = x1 - x2
    dy = y1 - y2
    length = dx/np.cos(angle)*width/xlimd
    dxslack = np.cos(angle)*slack/width*xlimd
    dyslack = np.sin(angle)*slack/height*ylimd
    u = dx - dxslack
    v = dy - dyslack
    if hideArrow is False:
        ax.quiver(x2, y2, u, v, scale_units='xy', angles = 'xy', scale = 1, width = 0.03*scale, units = 'inches')
    if marker is True:
        ax.plot(x, y,markerType, color = markerColor, markersize = markerSize, label = markerLabel)
    if label is not None:
        xt = x1 - dx + adjText[0]/width*xlimd
        yt = y1 - dy + adjText[1]/height*ylimd
        if np.abs(np.sin(angle)) > 0.5:
            ha = 'center'
        elif np.cos(angle) > 0.0:
            ha = 'right'
        else:
            ha = 'left'
        if np.sin(angle) > 0.5:
            va = 'top'
        elif np.sin(angle) < -0.5:
            va = 'bottom'
        else:
            va = 'center'
        ax.text(xt,yt,label,verticalalignment=va,horizontalalignment=ha,fontsize=fs)

def doublearrow(x1, x2, y, fig, ax, fs=8.0, slack=2.0, adjText=(0.0,0.0), label=None, scale=0.7):
    '''
	Make a double-ended horizontal arrow
	
	Parameters
	----------
    x1 : float
        x-axis point 1
    x2 : float
        x-axis point 2
    y : float
        y-axis height
    fig : figure object
        Figure to plot on
    ax : axis object
        Axis to plot on
    fs : float, optional
        Fontsize of label
    slack : float, optional
        Slack between points and arrow head
    adjText : tupple of floats, optional
        Adjust text if it crashes with arrow
    label : string
        Label over the arrow
        Use None for none
    scale : float, optional
        Scale of the arrow
	
	Returns
	-------
	None
	'''
    ylim = ax.get_ylim()
    ylimd = ylim[1] - ylim[0]
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    height = bbox.height*fig.dpi
    dx = (x2 - x1)/2.0#/width*xlimd
    cx = (x1 + x2)/2.0
    
    ax.quiver(cx, y, -dx, 0.0, scale_units='xy', angles = 'xy', scale = 1, width = 0.03*scale, units = 'inches')
    ax.quiver(cx, y, dx, 0.0, scale_units='xy', angles = 'xy', scale = 1, width = 0.03*scale, units = 'inches')
    
    if label is not None:
        xt = cx 
        yt = y + (2.5+adjText[1])/height*ylimd
        ha = 'center'
        va = 'bottom'
        ax.text(xt,yt,label,verticalalignment=va,horizontalalignment=ha,fontsize=fs)

def get_xy(x, y, line, ax):
    '''
	Get actual point of a line close to input point
	
	Parameters
	----------
    x : float
        x-axis point close to desired point
    y : float
        y-axis point close to desired point
    line : line object
        The line to search on
        Get from   line, = ax.plot()
    ax : axis object
        Axis to look on
	
	Returns
	-------
    xd : float
        Closest x-point
    yd : float
        Closest y-point
    I : int
        Data index of this point
	'''
    if type(line) == list:
        line = line[0]
    xlim = ax.get_xlim()
    xlimd = xlim[1] - xlim[0]
    ylim = ax.get_ylim()
    ylimd = ylim[1] - ylim[0]
    fig = ax.get_figure()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width = bbox.width
    height = bbox.height
    
    x = x/xlimd*width
    y = y/ylimd*height
    xd = np.array(line.get_xdata())/xlimd*width
    yd = np.array(line.get_ydata())/ylimd*height
    p = np.sqrt((x - xd)**2.0 + (y - yd)**2.0)
    I = np.argmin(p)
    return xd[I]*xlimd/width, yd[I]*ylimd/height, I

def subfiglabels(fig, fs, usetex, position='upperrightoutside', move=3.0, skipaxes=None, transpose=False, axes=None):
    '''
	Adds index to all subfigures in a placement of choice
    Should run fig.tight_layout() first
	
	Parameters
	----------
    fig : figure object
        Figure to add indices on
    fs : float
        Fontsize
    usetex : boolean
        Whether latex rendring should be used
    position : string, optional
        Position of indices:

        - upperleft
        - upperleftoutside
        - upperrightoutside
        - undercaption

    move : float, optional
        Move slightly if crashing with edges etc.
    transpose : bool, optional
        If true, subfiglabels are column-major instad
	
	Returns
	-------
	None
	'''
    if skipaxes == None:
        skipaxes = []
    if axes is None:
        axes = np.array(fig.get_axes())
        axes = axes.ravel()
    if transpose == True:
        gridspec = fig.axes[0].get_subplotspec().get_gridspec().get_geometry()
        axes = (np.reshape(axes, gridspec).T).ravel()
    if position == 'upperleft':
        for i in range(0, len(axes)):
            if i in skipaxes: continue
            ax = axes[i]
            bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            width = bbox.width*fig.dpi
            height = bbox.height*fig.dpi
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            xlimd = xlim[1] - xlim[0]
            ylimd = ylim[1] - ylim[0]
            
            x = xlim[0] + move/width*xlimd
            y = ylim[1] - move/height*ylimd
            if usetex is True:
                ax.text(x, y, r'\textbf{(%s)}' % (chr(i+97)), verticalalignment = 'top', horizontalalignment = 'left', fontsize=fs)
            else:
                ax.text(x, y, r'(%s)' % (chr(i+97)), verticalalignment = 'top', horizontalalignment = 'left', weight = 'bold', fontsize=fs)
    elif position == 'upperleftoutside':
        for i in range(0, len(axes)):
            if i in skipaxes: continue
            ax = axes[i]
            bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            width = bbox.width*fig.dpi
            height = bbox.height*fig.dpi
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            xlimd = xlim[1] - xlim[0]
            ylimd = ylim[1] - ylim[0]
            
            x = xlim[0] + move/width*xlimd
            y = ylim[1] + move/height*ylimd
            if usetex is True:
                ax.text(x, y, r'\textbf{(%s)}' % (chr(i+97)), verticalalignment = 'bottom', horizontalalignment = 'left', fontsize=fs)
            else:
                ax.text(x, y, r'(%s)' % (chr(i+97)), verticalalignment = 'bottom', horizontalalignment = 'left', weight = 'bold', fontsize=fs)
    elif position == 'upperrightoutside':
        for i in range(0, len(axes)):
            if i in skipaxes: continue
            ax = axes[i]
            bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            width = bbox.width*fig.dpi
            height = bbox.height*fig.dpi
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            xlimd = xlim[1] - xlim[0]
            ylimd = ylim[1] - ylim[0]
            
            x = xlim[1] + move/width*xlimd
            y = ylim[1] + move/height*ylimd
            if usetex is True:
                ax.text(x, y, r'\textbf{(%s)}' % (chr(i+97)), verticalalignment = 'bottom', horizontalalignment = 'right', fontsize=fs)
            else:
                ax.text(x, y, r'(%s)' % (chr(i+97)), verticalalignment = 'bottom', horizontalalignment = 'right', weight = 'bold', fontsize=fs)
    elif position == 'undercaption':
        for i in range(0, len(axes)):
            if i in skipaxes: continue
            ax = axes[i]
            bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            width = bbox.width*fig.dpi
            height = bbox.height*fig.dpi
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            xlimd = xlim[1] - xlim[0]
            ylimd = ylim[1] - ylim[0]
            xlbl_pos = ax.xaxis.get_label().get_window_extent().transformed(ax.transAxes.inverted())
            
            x = (xlim[1] + xlim[0])/2.0
            y = ylim[0] + xlbl_pos.y0*ylimd
            if usetex is True:
                ax.text(x, y, r'\textbf{(%s)}' % (chr(i+97)), verticalalignment = 'top', horizontalalignment = 'center', fontsize=fs)
            else:
                ax.text(x, y, r'(%s)' % (chr(i+97)), verticalalignment = 'top', horizontalalignment = 'center', weight = 'bold', fontsize=fs)
            
def useexp(ax, exp=True, thr=(-1,1)):
    '''
	Use exponential term on y-axis on an axis
	
	Parameters
	----------
    ax : axis object
        Axis to make exponential
    exp : boolean, optional
        Turn on or off exp.
    thr : tupple, optional
        Lower and upper 10^X threshold for using scientific notation
	
	Returns
	-------
	None
	'''
    if exp is True:
        ax.yaxis.get_major_formatter().set_powerlimits(thr)
    else:
        ax.yaxis.get_major_formatter().set_powerlimits((-99,99))

def plotharmonics(fhar, fsub, ax, **kwParameters):
    '''
	Plot harmonics and sidebands
	
	Parameters
	----------
    fhar : float
        Harmonic frequency
    fsub : float
        Sideband frequency
    ax : axis object
        Axis to plot on
    nhar : int or list of ints, optional
        How many harmonics to plot:

        - If integer, it defines how many harmonics to plot
        - If list, define which harmonics to plot, i.e. [1,2,4]
        - Else the axis xlim defines how many to plot

    nsub_p : int or list of ints, optional
        How many positive sidebands to plot:

        - If integer, it defines how many to plot for each harmonic
        - If list, it defines how many plot at each harmonic. i.e. [1,1,2]
        - Else 1 is plotted per harmonic

    nsub_n : int or list of ints, optional
        See description of nsub_p
    harcolor : string or tupple of floats, optional
        Color of each harmonic:

        - If string, normal colors like 'r' or 'k'
        - If tupple, RGBA is used. All numbers <0.0, 1.0>

    harstyle : string, optional
        Style of harmonic
    harwidth : float, optional
        Linewidth of harmonic
    harlabel : string, optional
        Legend label of harmonic
    subcolor : string or tupple of floats, optional
        Subband color:

        - If string, normal colors like 'r' or 'k'
        - If tupple, RGBA is used. All numbers <0.0, 1.0>

    substyle : string, optional
        Style of subbands
    subwidth : float, optional
        Linewidth of subbands
    sublabel : string, optional
        Legend label of subbands
	
	Returns
	-------
	None
	'''
    
    #Default arguments
    xlim = ax.get_xlim()
    nhar = int(np.floor(xlim[1]/fhar))
    if fsub == None:
        nsub_p = None
        nsub_n = None
    else:
        nsub_p = 1
        nsub_n = 1
    harcolor = (255.0/255.0, 215.0/255.0, 0.0/255.0, 1.0)
    harstyle = '-'
    harwidth = 2.0
    harlabel = None
    subcolor = (34.0/255.0, 177.0/255.0, 76.0/255.0, 1.0)
    substyle = '--'
    subwidth = 0.5
    sublabel = None
    
    #Check keywords
    if kwParameters is not None:
        for key, value in kwParameters.items():
            if key == 'nhar':
                nhar = value
            if key == 'nsub_p':
                nsub_p = value
            if key == 'nsub_n':
                nsub_n = value
            if key == 'harcolor':
                harcolor = value
            if key == 'harstyle':
                harstyle = value
            if key == 'harwidth':
                harwidth = value
            if key == 'harlabel':
                harlabel = value
            if key == 'subcolor':
                subcolor = value
            if key == 'substyle':
                substyle = value
            if key == 'subwidth':
                subwidth = value
            if key == 'sublabel':
                sublabel = value
    
    #Make values required
    if type(nhar) is not list:
        nhar = list(np.arange(1, nhar+1))
    if type(nsub_n) is not list and nsub_n is not None:
        nsub_n = list(np.ones(len(nhar), dtype = int)*nsub_n)
    if type(nsub_p) is not list and nsub_p is not None:
        nsub_p = list(np.ones(len(nhar), dtype = int)*nsub_p)
        
    harplotted = False
    subplotted = False
    ylim = ax.get_ylim()
    for i in range(0, len(nhar)):
        f = nhar[i]*fhar
        if harplotted is False and harlabel is not None:
            ax.plot([f, f], ylim, harstyle, color = harcolor, linewidth = harwidth, label = harlabel)
            harplotted = True
        else:
            ax.plot([f, f], ylim, harstyle, color = harcolor, linewidth = harwidth)
        if fsub is not None:
            for j in range(0, nsub_n[i]):
                fn = f - (j+1)*fsub
                if subplotted is False and sublabel is not None:
                    ax.plot([fn, fn], ylim, substyle, color = subcolor, linewidth = subwidth, label = sublabel)
                    subplotted = True
                else:
                    ax.plot([fn, fn], ylim, substyle, color = subcolor, linewidth = subwidth)
        if fsub is not None:
            for j in range(0, nsub_p[i]):
                fp = f + (j+1)*fsub
                if subplotted is False and sublabel is not None:
                    ax.plot([fp, fp], ylim, substyle, color = subcolor, linewidth = subwidth, label = sublabel)
                    subplotted = True
                else:
                    ax.plot([fp, fp], ylim, substyle, color = subcolor, linewidth = subwidth)
            
def plotfft(Y, df, ax, xlim=None, linetype='-k', width=None, label='', zorder=3, color=None):
    '''
	Plot a spectrum with certain limit in x and y direction
	
	Parameters
	----------
    Y : float 1D array
        Spectrum values
    df : float
        Delta frequency
    ax : axis object
        Axis to plot on
    xlim : None or array_like, optional
        X-axis limits:

        - None for showing the entire spectrum
        - [xmin, xmax] for a limit

    line : string, optional
        Design of the line, including color and linetype
    width : string, optional
        Linewidth
    label : string, optional
        Label for legends
    zorder : int, optional
        The plotting depth order
    color : string, or RGB/RGBA tupple
        The line color
        Will override line color
	
	Returns
	-------
    line : line object
        The spectrum line object
	'''
    
    if xlim is None:
        n0 = 0
        n1 = Y.size
    else:
        n0 = int(xlim[0]/df)
        n1 = int(xlim[1]/df + 1)
    if n1 > Y.size:
        n1 = Y.size
    line = ax.plot(np.arange(n0, n1)*df, Y[n0:n1], linetype, linewidth = width, label = label, zorder = zorder)
    if color is not None:
        line[0].set_color(color)
    return line

def plotcepstrum(cep, dt, ax, xlim=None, line='-k', width=None, label=None, zorder=3):
    '''
	Plot a cepstrum with certain limit in x and y direction
	
	Parameters
	----------
    cep : float 1D array
        Cepstrum values
    dt : float
        Delta time
    ax : axis object
        Axis to plot on
    xlim : None or array_like, optional
        X-axis limits:

        - None for showing the entire spectrum
        - [xmin, xmax] for a limit

    line : string, optional
        Design of the line, including color and linetype
    width : string, optional
        Linewidth
    label : string, optional
        Label for legends
	
	Returns
	-------
    line : line object
        The spectrum line object
	'''
    
    if xlim is None:
        n0 = 0
        n1 = cep.size
    else:
        n0 = int(xlim[0]/dt)
        n1 = int(xlim[1]/dt + 1)
    if n1 > cep.size:
        n1 = cep.size
    line = ax.plot(np.arange(n0, n1)*dt, cep[n0:n1], line, linewidth = width, label = label, zorder = zorder)
    return line
    
def plotbearingfaults(Y, df, X, bearing, maintitle='', harmonics=[8,8,8,8], subbands=2):
    '''
	Plots a 2x2 subplot figure of the same spectrum, but with harmonic lines
    and side-bands for each fault type
	
	Parameters
	----------
    Y : float 1D array
        Spectrum to plot
    df : float
        Delta frequency in Hz
    X : float
        Shaft speed in Hz
    bearing : float array_like
        Characteristic fault orders of bearing [inner,roller,cage,outer]
        Should be in orders (per revolution)
    maintitle : string, optional
        Main title of figure
    harmonics : array_like of ints, optional
        Number of harmonics to plot
    subbands : int, optional
        Number of side-bands to plot per harmonic
	
	Returns
	-------
	fig : figure object
        The figure
	'''
    
    fig, axes = figure(4, 1, width=500.0, height=600.0)
    for i in range(0, 4):
        if i == 0:
            title = 'Inner Ring'
            sideband = 1.0
        elif i == 1:
            title = 'Rollers'
            sideband = bearing[2]
        elif i == 2:
            title = 'Cage'
            sideband = None
        elif i == 3:
            title = 'Outer Ring'
            sideband = None
        ax = axes[i]   
        xlim = [0.0, (harmonics[i] + 1)*bearing[i]]
        plotfft(Y, df/(X), ax, xlim=xlim, width = None, linetype = '-k')
        #Find appropriate ylim
        i1 = int(0.15/(df/X))
        i2 = int(xlim[1]/(df/X))
        ax.set_ylim([0, np.max(Y[i1:i2])*1.15])
        ax.set_xlim(xlim)
        ax.set_title(title)
        plotharmonics(bearing[i], sideband, ax, nsub_n = subbands, nsub_p = subbands)
    
    fig.suptitle(maintitle)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    return fig

def set_xylim(xlim, ax, ylim_mult=[1.0, 1.1], lines=None):
    '''
    Set the desired x-lim, and adjustts the y-lim to fit the new x-lim

    Parameters
    ----------
    xlim : float 1D array_lime
        The desired x-lim
    ax : plt axis
        The axis
    ylim_mul : float 1D array_like, optional
        Margins for the calculated y-limit in gain
    lines : 1D array_like with lines, optional
        Choose to get y-lim of certain lines instad of all available in ax.
        None if all should be used.
    '''

    lines = ax.get_lines()
    ymax = []
    ymin = []
    for line in lines:
        x, y = line.get_data()
        i = np.where( (x >= xlim[0]) &  (x <= xlim[1]) )[0]
        if len(i) > 1:
            ymax.append(y[i].max())
            ymin.append(y[i].min())
    ymax = np.array(ymax)
    ymin = np.array(ymin)
    ax.set_xlim(xlim)
    ax.set_ylim(ymin.min()*ylim_mult[0], ymax.max()*ylim_mult[1])

def set_xticks(ax, majorspacing, minorspacing=None):
    '''
    Set the major and minor spacing of the x-axis

    Parameters
    ----------
    ax : figure axis
        The axis
    majorspacing : float or array_like
        The desired major spacing.
        If single value: Spaces at multiples of majorspacing
        If array_like: Makes spacing at these points
    minorspacing : float or array_like, optional
        Sets the minor spacing. Same rules as majorspacing
    '''

    if type(majorspacing) is float:
        majorLocator = MultipleLocator(majorspacing)
        majorFormatter = FormatStrFormatter('%f')
    elif type(majorspacing) is int:
        majorLocator = MultipleLocator(majorspacing)
        majorFormatter = FormatStrFormatter('%d')
    else:
        majorLocator = majorspacing
        if majorspacing[0] == int(majorspacing[0]):
            majorFormatter = FormatStrFormatter('%d')
        else:
            majorFormatter = FormatStrFormatter('%f')

    if type(minorspacing) is float:
        minorLocator = MultipleLocator(minorspacing)
    elif type(minorspacing) is int:
        minorLocator = MultipleLocator(minorspacing)
    else:
        minorLocator = minorspacing

    ax.xaxis.set_major_locator(majorLocator)
    # ax.xaxis.set_major_formatter(majorFormatter)
    ax.xaxis.set_minor_locator(minorLocator)

def set_yticks(ax, majorspacing, minorspacing=None):
    '''
    Set the major and minor spacing of the y-axis

    Parameters
    ----------
    ax : figure axis
        The axis
    majorspacing : float or array_like
        The desired major spacing.
        If single value: Spaces at multiples of majorspacing
        If array_like: Makes spacing at these points
    minorspacing : float or array_like, optional
        Sets the minor spacing. Same rules as majorspacing
    '''

    # Major formatter
    if type(majorspacing) is float:
        majorLocator = MultipleLocator(majorspacing)
        majorFormatter = FormatStrFormatter('%f')
    elif type(majorspacing) is int:
        majorLocator = MultipleLocator(majorspacing)
        majorFormatter = FormatStrFormatter('%d')
    else:
        majorLocator = majorspacing
        if majorspacing[0] == int(majorspacing[0]):
            majorFormatter = FormatStrFormatter('%d')
        else:
            majorFormatter = FormatStrFormatter('%f')
    ax.yaxis.set_major_locator(majorLocator)

    # Minor formatter
    if minorspacing is not None:
        if type(minorspacing) is float:
            minorLocator = MultipleLocator(minorspacing)
        elif type(minorspacing) is int:
            minorLocator = MultipleLocator(minorspacing)
        else:
            minorLocator = minorspacing
        ax.yaxis.set_minor_locator(minorLocator)

def plotforcingfreqs(Y, df, X, forcingfreqs, forcingfreqtitles=None, maintitle = '', harmonics = 8):
    '''
	Plots a 2x2 subplot figure of the same spectrum, but with harmonic lines
    and side-bands for each fault type
	
	Parameters
	----------
    Y : float 1D array
        Spectrum to plot
    df : float
        Delta frequency in Hz
    X : float
        Shaft speed in Hz
    forcingfreqs : float array_like
        Forcing frequencies
    forcingfreqtitles : list of strings, optional
        Title for each forcing frequency
        Must be as long as len(forcingfreqs)
    maintitle : string, optional
        Main title of figure
    harmonics : int, optional
        Number of harmonics to plot
    subbands : int, optional
        Number of side-bands to plot per harmonic
	
	Returns
	-------
	fig : figure object
        The figure
	'''
    
    N = len(forcingfreqs)
    fig, axes = figure(N,1, width = 500.0, height = 600.0/4*N)
    if N == 1:
        axes = np.array([axes,])
    for i in range(0, N):
        if forcingfreqtitles is not None:
            title = forcingfreqtitles[i]
        sideband = None
        ax = axes[i]   
        xlim = [0.0, (harmonics + 1)*forcingfreqs[i]]
        plotfft(Y, df/(X), ax, xlim=xlim, width = None, linetype = '-k')
        #Find appropriate ylim
        i1 = int(0.15/(df/X))
        i2 = int(xlim[1]/(df/X))
        ax.set_ylim([0, np.max(Y[i1:i2])*1.15])
        ax.set_xlim(xlim)
        ax.set_title(title)
        plotharmonics(forcingfreqs[i], sideband, ax, nsub_n = 1, nsub_p = 1)
    
    fig.suptitle(maintitle)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    return fig

def get_decimatedplotvalues(t, vib, r=None, p=None):
    '''
	Get a decimated version of the input signal where peaks are visibly intact
	
	Parameters
	----------
    t : float 1D array
        Time signal
    vib : float 1D array
        Signal
    r : int, optional
        Denominator of fraction to use. Larger than 1
        Either r or p must be supplied
    p : int, optional
        Number of points to keep
        Either r or p must be supplied
	
	Returns
	-------
	tdec : float 1D array
        Decimated time signal
    vibdec : float 1D array
        Decimated signal
	'''
    
    assert p is not None or r is not None
    if r is not None:
        r = int(r)
        r = r*2
    else:
        if vib.size <= p:
            r = 1
            return t, vib
        else:
            r = int(vib.size/p)
    N = int(np.ceil(float(vib.size)/float(r)))
    vib2 = np.zeros(2*N)
    tdec = np.zeros(vib2.size)
    for i in range(0, N):
        Min = np.min(vib[i*r:(i+1)*r])
        Max = np.max(vib[i*r:(i+1)*r])
        vib2[2*i] = Min
        vib2[2*i+1] = Max
        tdec[2*i] = t[i*r]
        tdec[2*i + 1] = t[i*r]
    return tdec, vib2

def decimatedplot(t, vib, p, ax, direct=False):
    '''
	Plot a decimated version where peaks are intact
	
	Parameters
	----------
    t : float 1D array
        Time signal
    vib : float 1D array
        Signal
    p : int
        Number of points to keep (roughly)
    ax : axis
        Axis to plot on
    direct : bool
        If direct decimation should be performed instead of
        preserving peaks
	'''
    if direct is False:
        if vib.size <= p:
            r = 1
        else:
            r = int(vib.size/p)
        tdec, vibdec = get_decimatedplot(t, vib, r=r)
    else:
        tdec = np.linspace(t[0], t[-1], p)
        f = interp1d(t, vib)
        vibdec = f(tdec)
    ax.plot(tdec, vibdec)

def filterbankvisualization(filters, level, ymax=1.03):
    '''
	Visualizes the effect of filters in a filterbank.
    Can also be used to see the frequency response of a single filter
	
	Parameters
	----------
    filters : list of 1D arrays
        List of filter kernels
    level : int
        The filterbank level. 0 means one filtering process
    ymax : float, optional
        The maximum y-axis value for plot. Use None to see all.
	
	Returns
	-------
	fig : figure object
        The figure
	'''
    
    w = np.linspace(-np.pi, np.pi, 2**18)
    r = []
    lines = ['r','b','g','c','m','k']
    d = len(filters)
    for i in range(0, d):
        w, temp = freqz(filters[i], a = 1, worN=w)
        r.append(temp)
    fig, ax = plt.subplots(level+1, 1, figsize = (16,(level+1)*1.5+4))
    if level == 0:
        for i in range(0, d):
            ax.plot(w/(2*np.pi),np.abs(r[i]),lines[i%6], label='signal{}'.format(i))
        ax.set_xticks(np.arange(-0.5, 0.6, 0.1))
        ax.set_xlim([-0.5, 0.5])
        if ymax is not None:
            ax.set_ylim([0.0, ymax])
    else:   
        for i in range(0, d):
            ax[0].plot(w/(2*np.pi),np.abs(r[i]),lines[i%6])
        ax[0].set_xticks(np.arange(-0.5, 0.6, 0.1))
        ax[0].set_xlim([-0.5, 0.5])
        ax[0].set_ylim([0.0, 1.03])
    
    xfilt = deepcopy(r)
    for i in range(0, level):
        xfiltNew = []
        for j in range(0, len(xfilt)):
            for rk in range(0, d):
                rd = downsample(r[rk],2**(i+1))
                rdl = rd.size//2
                
                x = np.zeros(w.size, dtype = 'complex')
                if j % 2 == 0:
                    for k in range(0, 2**(i+1)+1):
                        if k == 0:
                            x[0:rdl] = rd[rdl:]
                        elif k == 2**(i+1):
                            x[-rdl:] = rd[:rdl]
                        else:
                            x[rdl+(k-1)*2*rdl:rdl+(k)*2*rdl] = rd
                else:
                    for k in range(0, 2**(i+1)):
                        x[k*2*rdl:(k+1)*2*rdl] = rd
                    
                x = x*xfilt[j]
                ax[i+1].plot(w/(2*np.pi), np.abs(x), lines[rk%6], label='signal{}'.format(rk) if i == 0 else None)
                xfiltNew.append(np.copy(x))
                
        xfilt = deepcopy(xfiltNew)
        ax[i+1].set_xticks(np.arange(-0.5, 0.6, 0.1))
        ax[i+1].set_xlim([-0.5, 0.5])
        if ymax is not None:
            ax[i+1].set_ylim([0.0, ymax])
    return fig, ax

def plotsquare(corner1, corner2, ax, line='-r', label='', linewidth=None):
    '''
	Plot a square
	
	Parameters
	----------
    corner1 : tupple or array_like of floats
        (x,y) position of lower left corner
    corner2 : tupple or array_like of floats
        (x,y) position of upper right corner
    ax : axis object
        Axis to plot on
    line : string, optional
        Design of line with color and linetype
    label : string, optional
        Legend label
	
	Returns
	-------
	None
	'''
         
    x1 = corner1[0]
    x2 = corner2[0]
    y1 = corner1[1] 
    y2 = corner2[1]
    x = np.array([x1, x2, x2, x1, x1])
    y = np.array([y1, y1, y2, y2, y1])
    ax.plot(x, y, line, linewidth=linewidth)
    
class get_xydata():
    '''
	Get closest point on graph by clicking on the figure
	
	Parameters
	----------
	fig : figure object
        Figure to listen on
    ax : axis object
        Axis to get values from
    line : line object
        Line to snap on
    snap : boolean, optional
        Whether closest point on line should be returned
    label: string, optional
        For figures with multiple axes, the label can help distinguish them.

	Returns
	-------
	None
	'''

    def __init__(self, fig, ax, line, snap = True, label=None):
        self.xd = []
        self.yd = []
        self.cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid2 = fig.canvas.mpl_connect('button_release_event', self.onrelease)
        self.cid3 = fig.canvas.mpl_connect('key_press_event', self.onkeypress)
        self.ax = ax
        self.line = line
        self.snap = snap
        self.label = label
        self.finished = False
        self.time_press = time.time()
        self.time_release = 0.0

    def onkeypress(self, event):
        if event.key.capitalize() == 'A':
            self.finished = True

    def onclick(self, event):
        self.time_press = time.time()
    
    def onrelease(self, event):
        self.time_release = time.time()
        if self.time_release - self.time_press < 0.3:
            if event.button == 1:
                if self.snap:
                    x, y, I = get_xy(event.xdata, event.ydata, self.line, self.ax)
                    self.xd.append(deepcopy(x))
                    self.yd.append(deepcopy(y))
                else:
                    self.xd.append(deepcopy(event.xdata))
                    self.yd.append(deepcopy(event.ydata))
            elif event.button == 3:
                self.xd = self.xd[0:-1]
                self.yd = self.yd[0:-1]
            self.printdata()
            
    def printdata(self):
        str1 = '    xd = np.array(['
        for i in range(0, len(self.xd)):
            str1 += str(self.xd[i]) + ', '
        str1 += '])'
        str2 = '    yd = np.array(['
        for i in range(0, len(self.yd)):    
            str2 += str(self.yd[i]) + ', '
        str2 += '])'
        print('\n')
        if self.label is not None:
            print(self.label)
        print(str1)
        print(str2)

def rgb2hex(r, g, b):
    '''
    Convert RGB to a hex string

    Parameters
    ----------
    r : int
        Red <0, 255>
    g : int
        Green <0, 255>
    b : int
        Blue <0, 255>

    Returns
    -------
    hex : string
        Hex string
    '''
    # Assume all are ints 0 to 255
    if type(r) == float:
        r, g, b = np.array(np.array([r, g, b])*255, int)
    h = '#'
    for e in (r, g, b):
        s = hex(e)
        if len(s) == 3:
            h += '0%s' % (s[2])
        else:
            h += s[2:4]
    return h

class PlotUpdater():
    '''
    Add dynamic switching of graphs by changing data
    Hold shift and:
    - right: increase index
    - left: decrease index
    - up: last index
    - down: first index

    Parameters
    ----------
    fig : plt figure
        Figure
    '''

    def __init__(self, fig):
        self.xd = []
        self.yd = []
        self.listener = fig.canvas.mpl_connect('key_press_event', self._onkeypress)
        self.lines = []
        self.lineDatas = []
        self.indexes = []
        self.axes = []
        self.fig = fig

    def addLine(self, line, ax, lineData):
        '''
        Add a line to the figure

        Parameters
        ----------
        line : plt line pbject
            The line object to add
        ax : axis
            Axis object the line belongs to
        lineData : dictionary or list of dictionaries
            Dictionary must contain 'x' for x-data and
            'y' for y-data
        '''
        self.lines.append(line)
        if type(lineData) == list:
            self.lineDatas.append(lineData)
            self.indexes.append(len(lineData) - 1)
        else:
            self.lineDatas.append([lineData,])
            self.indexes.append(0)
        self.axes.append(ax)

    def _onkeypress(self, event):
        if event.key == 'shift+right':
            for i in range(len(self.indexes)):
                self.indexes[i] += 1
                if self.indexes[i] == len(self.lineDatas[i]):
                    self.indexes[i] -= 1
            self._updateFigure()
        elif event.key == 'shift+left':
            for i in range(len(self.indexes)):
                self.indexes[i] -= 1
                if self.indexes[i] == -1:
                    self.indexes[i] += 1
            self._updateFigure()
        elif event.key == 'shift+up':
            for i in range(len(self.indexes)):
                self.indexes[i] = len(self.lineDatas[i]) - 1
            self._updateFigure()
        elif event.key == 'shift+down':
            for i in range(len(self.indexes)):
                self.indexes[i] = 0
            self._updateFigure()

    def _updateFigure(self):
        temp = []
        for ax in self.axes:
            if not ax in temp:
                temp.append(ax)
                ax.draw_artist(ax.patch)
        for i, ax in enumerate(self.axes):
            line = self.lines[i]
            lineData = self.lineDatas[i][self.indexes[i]]
            line.set_xdata(lineData['x'])
            line.set_ydata(lineData['y'])
            ax.draw_artist(line)

        self.fig.canvas.update()
        self.fig.canvas.flush_events()

def get_allLinesAndLabels(axes):
    '''
    Get all lines and labels in a list of axes

    Parameters
    ----------
    axes : list of axis
        Axes to get lines and labels from

    Returns
    -------
    lines : list of line
        Identified lines
    labels : list of label
        Line labels
    '''

    lines = []
    labels = []
    for ax in axes:
        linesInAx = ax.get_lines()
        for line in linesInAx:
            label = line.get_label()
            if label[0] != '_':
                lines.append(line)
                labels.append(label)
        for patch in ax.patches:
            label = patch.get_label()
            if label[0] != '_':
                lines.append(patch)
                labels.append(label)
    return lines, labels

def alignYLabels(fig, axes):
    '''
    Align y labels to a single line for the axes
    The left-most positioned label is used

    Parameters
    fig : figure
        Figure which axes are on
    axes : list of axes
        Axes to align
    '''

    ax = axes[0]
    fig.canvas.draw()
    temp = ax.axes.yaxis.label.get_window_extent()
    x0_1 = temp.x0
    ax.yaxis.labelpad += 1
    fig.canvas.draw()
    temp = ax.axes.yaxis.label.get_window_extent()
    x0_2 = temp.x0
    ax.yaxis.labelpad -= 1
    dx0 = x0_2 - x0_1
    # Find leftmost label
    smallestx0 = 1e15
    for i in range(0, len(axes)):
        ax = axes[i]
        temp = ax.axes.yaxis.label.get_window_extent()
        x0 = temp.x0
        if x0 < smallestx0:
            smallestx0 = x0

    # Adjust labels
    for i in range(0, len(axes)):
        ax = axes[i]
        temp = ax.axes.yaxis.label.get_window_extent()
        x0 = temp.x0
        ax.yaxis.labelpad += (smallestx0 - x0)/dx0
    fig.canvas.draw()

def set_tickformat(ax, axes, fmat='%.1f'):
    '''
    Set certain format on numbers on axes

    Parameters
    ----------
    ax : axis object
        Axis
    axes : string
        both, x or y
    fmat : string
        Number format to use
    '''

    if axes == 'both':
        ax.xaxis.set_major_formatter(FormatStrFormatter(fmat))
        ax.yaxis.set_major_formatter(FormatStrFormatter(fmat))
    elif axes == 'x':
        ax.xaxis.set_major_formatter(FormatStrFormatter(fmat))
    elif axes == 'y':
        ax.yaxis.set_major_formatter(FormatStrFormatter(fmat))
    else:
        raise Exception('Axes object does not exist')

def textCorner(fig, ax, label, textbbox=None):
    '''
    Add text in upper left corner of axis

    Parameters
    ----------
    fig : figure object
    ax : axis object
    label : string
        Label in the corner
    textbbow : bbox
        Text bounding box
        e.g. bboxMe = {'facecolor':'white', 'edgecolor':'white', 'zorder':-100, 'pad':0}
    '''

    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width = bbox.width*fig.dpi
    height = bbox.height*fig.dpi
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xlimd = xlim[1] - xlim[0]
    ylimd = ylim[1] - ylim[0]
    x = xlim[0] + 0.03*xlimd
    y = ylim[1] - 0.03*ylimd
    ax.text(x, y, r'%s' % (label), verticalalignment='top', horizontalalignment='left', bbox=textbbox)

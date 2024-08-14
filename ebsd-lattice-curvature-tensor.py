#######################################################################
# Functions for extracting lattice curvature tensor from orientation
# fields. Written for EBSD datasets, but may be generizable to other
# orientation fields.
#
# Originally written by: Evan Musterman
# ejm218@lehigh.edu, emusterma@bnl.gov
# 2024-08-14
# 
# Based on:
# Pantelon, W. "Resolving the geometrically necessary dislocation
# content by conventional electron backscattering diffraction",
# Scripta Materialia 58, 994-997 (2008).
# He, W.; Ma, W.; Pantleon, W.; "Microstructure of individual
# grains in cold-rolled aluminum from orientation inhomogeneities
# resolved by electron backscattering diffraction", Materials
# Science and Engineering A 494, 21-27 (2008).
#
# Pantleon et al. took the lattice curvature one step furhter to
# resolve geometrically necessary dislocations. They also used
# quaternions which are generally more efficient. This implementation 
# stops at just the lattice curvature and uses rotation matrices and
# could thus be improved.
#
#######################################################################

import numpy as np
from scipy.spatial import KDTree
from scipy.stats import linregress, t


# Convenience function to convert lists of Euler Angles in
# the passive Bunge definition into a list orientation (g)
# matrices
def g_func(phi1,
           phi,
           phi2,
           rad=False):

    '''
    phi1        (float) First Euler angle in Bunge definition
    phi         (float) Second Euler angle in Bunge definition
    phi2        (float) Third Euler angle in Bunge definition
    rad         (bool)  True if angles given in radians
    '''

    # Check and convert input angles
    # Assumes some random pixel will exceed threshold in dataset
    if rad & (np.max([phi1, phi, phi2]) > 30): 
        raise ValueError('Euler angle values in radians exceed expected bounds!')
    elif not rad:
        phi1 = np.radians(phi1)
        phi = np.radians(phi)
        phi2 = np.radians(phi2)
        
    # Just a bit of shorthand
    from numpy import cos as c
    from numpy import sin as s

    # Passive Bunge definition
    g_T =   ([c(phi1) * c(phi2) - s(phi1) * s(phi2) * c(phi),
              s(phi1) * c(phi2) + c(phi1) * s(phi2) * c(phi),
              s(phi2) * s(phi)],
            [-c(phi1) * s(phi2) - s(phi1) * c(phi2) * c(phi),
             -s(phi1) * s(phi2) + c(phi1) * c(phi2) * c(phi),
             c(phi2) * s(phi)],
            [s(phi1) * s(phi),
             -c(phi1) * s(phi),
             c(phi)])
    g_T = np.asarray(g_T).T

    # Transpose array to be correct.
    g = np.array([g_T[i].T for i in range(len(g_T))])

    return g


# Function to estimate curvature tensor
def curvature_tensor(x,
                     y,
                     g,
                     con_int=0.95,
                     k_max=10,
                     window=2.5,
                     IQ=None,
                     CI=None,
                     FIT=None,
                     norm_iq=[],
                     ci=[],
                     fit=[]):

    '''
    x       (list)  List or array of x, [010],
                    2-axis spatial coordinates in microns
    y       (list)  List or array of y, [100],
                    1-axis spatial coordinates in microns
    g       (list)  List of orientation matrices. Passive Bunge definition
    con_int (float) Confidence interval of curvature measurement as fraction.
                    Default is 0.95.
    k_max   (float) Maximum considered curvature value in deg/micron.
                    Higher values are considered error and ignored.
                    Default is 10.
    window  (float) Window in microns around pixel used to calculate
                    the local curvature.
                    Default is 2.5 microns, but should be higher with
                    bigger step sizes
    IQ      (float) Cut-off fraction of normalized image quality (IQ)
                    values to use.
                    (e.g., 0.4 uses top 60% of values)
    CI      (float) Cut-off value of confidence index (CI) values to use.
                    (e.g., 0.1 uses all higher)
    FIT     (float) Cut-off value of fit values to use.
                    (e.g., 2 uses all lower fits)
    norm_iq (list)  Normalized image quality values per pixel
    ci      (list)  Confidence index values per pixel
    fit     (list)  Fit values per pixel
    '''

    # Check to make sure filters have values to parse
    if ((None != (IQ or CI or FIT)) and
       (0 == (len(norm_iq) or len(ci) or len(fit)))):
        raise ValueError('''You are trying to filter by IQ, CI, 
                            or Fit without defining these values!''')
    
    # Check curvature sampling criterion to see if default
    if any([con_int == 0.95, k_max == 10, window == 2.5]):
        print('Using defualt curvature calculation criteria:')
        print(f'\t{window} μm window for calculating curvature.')
        print(f'''\t{k_max}°/μm maximum allowable curvature. 
                Higher values considered error and ignored.''')
        print(f'\tReporting {con_int * 100}% confidence index.')

    # Calculating curvature tensor components for every pixel
    # based on nearest neighbors and curvature regressions
    nan = float('NaN')
    points = np.array([x, y]).T
    tree = KDTree(points)
    k_map, std_map = [], []
    meta = {'Window':window,
            'k_max':k_max,
            'Confidence Index':con_int,
            'IQ':IQ,
            'CI':CI,
            'FIT':FIT}

    for j in range(len(g)):
        # Cuvature components
        k11, k12, k13, k21, k22, k23, k31, k32, k33 = (nan,) * 9
        # Standard error of slope fits
        std11, std21, std31, std12, std22, std32 = (nan,) * 6
        
        if ((IQ == None or norm_iq[j] > IQ) and
            (CI == None or ci[j] > CI) and
            (FIT == None or fit[j] < FIT)):
            
            idx = tree.query_ball_point(points[j], window)
            xi, yi, w1i, w2i, w3i = [], [], [], [], []
            
            # Rotations are in radians at this point
            for i in idx:
                dg = np.dot(g[j], g[i].T)
                w_mag = np.arccos(0.5 * (dg[0][0] + 
                                         dg[1][1] + 
                                         dg[2][2] - 1))
                
                factor = w_mag / np.sin(w_mag)
                if w_mag == 0:
                    w1, w2, w3 = 0, 0, 0
                else:
                    w1 = 0.5 * (dg[1][2] - dg[2][1]) * factor #23-32
                    w2 = 0.5 * (dg[2][0] - dg[0][2]) * factor #31-13
                    w3 = 0.5 * (dg[0][1] - dg[1][0]) * factor #12-21
                # Rotation in crystal coordinates
                wc = np.array(([w1, w2, w3]))
                # Rotation in sample coordinates
                ws = np.dot(g[j].T, wc)
                w1, w2, w3 = ws[0], ws[1], ws[2]
                
                # Cut-off misorientation; attempting to 
                # ignore misindexed pixels
                if np.any(np.abs([w1, w2, w3]) > 
                          np.radians(k_max * window)):
                    w1, w2, w3 = nan, nan, nan
                
                #Eliminating contribution of pixels outside of filter
                if ((IQ != None and norm_iq[i] < IQ) or
                    (CI != None and ci[i] < CI) or
                    (FIT != None and fit[i] > FIT)):
                    w1, w2, w3 = nan, nan, nan
                w1i.append(w1)
                w2i.append(w2)
                w3i.append(w3)
                xi.append(x[i])
                yi.append(y[i])

            w1i = np.asarray(w1i)
            w2i = np.asarray(w2i)
            w3i = np.asarray(w3i)
            xi, yi = np.asarray(xi), np.asarray(yi)
            mask1 = ~np.isnan(w1i)
            mask2 = ~np.isnan(w2i)
            mask3 = ~np.isnan(w3i)

            # Slope fits and confidence interval calculations
            if len(np.unique(yi[mask1])) > 1:
                k11, bi, r11, pi, se = linregress(yi[mask1], w1i[mask1])
                std11 = se * t.ppf((1 + con_int) / 2, np.sum(mask1) - 2)
            if len(np.unique(yi[mask2])) > 1:
                k21, bi, r21, pi, se = linregress(yi[mask2], w2i[mask2])
                std21 = se * t.ppf((1 + con_int) / 2, np.sum(mask2) - 2)
            if len(np.unique(yi[mask3])) > 1:
                k31, bi, r31, pi, se = linregress(yi[mask3], w3i[mask3])
                std31 = se * t.ppf((1 + con_int) / 2, np.sum(mask3) - 2)
            if len(np.unique(xi[mask1])) > 1:
                k12, bi, r12, pi, se = linregress(xi[mask1], w1i[mask1])
                std12 = se * t.ppf((1 + con_int) / 2,np.sum(mask1) - 2)
            if len(np.unique(xi[mask2])) > 1:
                k22, bi, r22, pi, se = linregress(xi[mask2], w2i[mask2])
                std22 = se * t.ppf((1 + con_int) / 2, np.sum(mask2) - 2)
            if len(np.unique(xi[mask3])) > 1:
                k32, bi, r32, pi, se = linregress(xi[mask3], w3i[mask3])
                std32 = se * t.ppf((1 + con_int) / 2, np.sum(mask3) - 2)
        
        k = np.array(([k11, k12, k13], [k21, k22, k23], [k31, k32, k33]))
        std = np.array([std11, std21, std31, std12, std22, std32])
        std[std == np.inf] = nan
        std_map.append(std)
        k_map.append(k)

    k_map = np.asarray(k_map)
    std_map = np.asarray(std_map)

    # Convert to degrees
    k11 = k_map[:, 0, 0] * (180 / np.pi)
    k21 = k_map[:, 1, 0] * (180 / np.pi)
    k31 = k_map[:, 2, 0] * (180 / np.pi)
    k12 = k_map[:, 0, 1] * (180 / np.pi)
    k22 = k_map[:, 1, 1] * (180 / np.pi)
    k32 = k_map[:, 2, 1] * (180 / np.pi)
    k_map = np.asarray([k11, k21, k31, k12, k22, k32]).T

    return k_map, std_map, meta
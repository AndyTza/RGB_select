import astropy.units as u
from dustmaps.sfd import SFDWebQuery
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.io import ascii, fits


# author: Anastasios (Andy) Tzanidakis
def dusty(data):
    """For 2MASS bandpasses: ks, j and h magnitudes, dusty will return the corrected magnitudes based of the
    SFD E(B-V) values at a given line of sight

    Input
    ------
    data(astropy.table): Astropy table contaning (contains: ks_mag, j_mag, h_mag)

    Output
    -------
    data(astropy.table): Initial data table in addition to new columns: (ks0, j0, h0, aks, aj, ah)


    """

# Choose R constant for exctinction in proper bandpass Yuan et al. 2013 values
    r_ks = 0.306 # ks-band
    r_j = 0.72 # j-band
    r_h = 0.46 # h-band

    l, b = data['l'], data['b']

    N = len(l) # number of stars in the data file
    query_max = 1000000 # maximum number of sources we can query
    N_int = N/query_max # how many times are we going to need to split the data
    L = np.arange(0, (int(N_int)+1)*query_max, step=query_max) # steps
    L = np.append(L, N) # add last value of total N points

    sfd = SFDWebQuery() # call SFD query function

    ks0 = np.ndarray(shape=(N,1)) # empty array for corrected magnitudes
    j0 = np.ndarray(shape=(N,1)) # empty array for corrected magnitudes
    h0 = np.ndarray(shape=(N,1)) # empty array for corrected magnitudes
    aks = np.ndarray(shape=(N,1)) # empty array for corrected magnitudes
    aj = np.ndarray(shape=(N,1)) # empty array for corrected magnitudes
    ah = np.ndarray(shape=(N,1)) # empty array for corrected magnitudes

    for i in range (0, len(L)-1):

        # adjusted coordinates based on step-size
        l_ad = l[L[i]:L[i+1]]
        b_ad = b[L[i]:L[i+1]]
        coords = SkyCoord(l_ad*u.deg, b_ad*u.deg, frame='galactic')

        # Query coords and return E(B-V) SFD for each source
        ebv_sfd = sfd(coords)

        # correct each magnitude by the SFD
        corr_ks = ((data['ks_mag'])[L[i]:L[i+1]]) - (r_ks*ebv_sfd)
        corr_j = ((data['j_mag'])[L[i]:L[i+1]]) - (r_j*ebv_sfd)
        corr_h = ((data['h_mag'])[L[i]:L[i+1]]) - (r_h*ebv_sfd)

        # corrected magnitudes
        ks0[L[i]:L[i+1], 0] = corr_ks
        j0[L[i]:L[i+1], 0] = corr_j
        h0[L[i]:L[i+1], 0] = corr_h
        # exctinction coefficient appends
        aks[L[i]:L[i+1], 0] = r_ks*ebv_sfd
        aj[L[i]:L[i+1], 0] = r_j*ebv_sfd
        ah[L[i]:L[i+1], 0] = r_h*ebv_sfd


    dat = Table([ks0, j0, h0, aks, aj, ah], names=('ks0', 'j0', 'h0', 'A_ks', 'A_j', 'A_h'))

    data.add_columns([dat['ks0'], dat['j0'], dat['h0'], dat['A_ks'], dat['A_j'], dat['A_h']])

    return (data)


# This pipeline will take the astropy table with corrected magnitudes and it will
# select M-giat stars given properties basedf on J-K etc colors!

def Mgy(table, JKs0=0.85, up_ks0=9.5, down_ks0=12.5):

    """M-giant selections by Sharma et al. 2010"""

    # Ks0 < 12 magnitude
    # (J-Ks)0 > 0.97
    # (J-H)0 < 0.561*(J-Ks)0 + 0.36
    # (J-H)0 > 0.561*(J-Ks) + 0.19

    # first we're going to estimate the colors: (J-Ks)0 & (J-H)0
    jks0 = table['j0'] - table['ks0'] # j0-ks0
    jh0 = table['j0'] - table['h0'] # j0-h0

    # Photometric cuts suggested by Sharama et al. 2010 for selecting M-giant stars
    giant = np.where((table['ks0']>up_ks0) & (table['ks0']<down_ks0) & (jks0>JKs0) & (jh0<(0.561*jks0+0.36)) & (jh0>(0.561*jks0+0.19)))

    print (giant[0])

    return table[giant[0]]

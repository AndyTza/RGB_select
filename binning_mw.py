import numpy as np
from astropy.table import Table

# Author: Anastasios (Andy) Tzanidakis
# Contact: at3156@columbia.edu

class galactic_binner():
    def __init__(self, distance, l, b):
        """
        Input:
        ------
        distance: distance [kpc] (float)
        l: Galactic Longtitude [deg] (float)
        b: Galactic Lattitude [deg] (float)
        """
        self.distance = distance
        self.l = l
        self.b = b

    def distance_binner(self, min_dist=0, max_dist=40, dr=1, breath_factor=0.5, star_limit=50):
        """
        This function will take a given distance distribution and bin the data, such that
        in each bin there is a minimum number of stars per bin

        Input:
        ------
        distance: distance of data (float)
        min_dist: minimum distance in bin scanning (float)
        max_dist: maximum distance in bin scanning (float)
        dr: distance bin length (float)
        breath_factor: value by which the distance bin will expand (float)
        star_limit: minimum number of stars per bin (float)

        Output:
        -------
        N_bin: Number of stars in each scanned distance bin (astropy.Table)
        dist_scan: Scanned distance over which we estimated N_bin (astropy.Table)
        del_r: distance bin length (astropy.Table)
        """
        N = len(self.distance) # total number of data points
        bin_scanner = np.arange(min_dist, max_dist, step=dr) # Arange bins based on the the number of steps dr

        # Empty lists
        N_bin, distance_scan, del_r = [], [], []

        for i in range(0, len(bin_scanner)-1):
            # Min, Max distance in each bin
            scan_min, scan_max = bin_scanner[i], bin_scanner[i+1]
            find = np.where((self.distance>scan_min) & (self.distance<scan_max)) # stars between min,max distance bin
            N_stars = len(self.distance[find]) # Number of stars in the ith dmin, dmax bin

            if N_stars>=star_limit:
                N_bin.append(N_stars)
                distance_scan.append(scan_max)
                del_r.append(dr)

            # If the number of stars in that bin are not greater than 50, begin to expand bin-size
            while N_stars<=star_limit:
                scan_max = scan_max + breath_factor
                if scan_max > max_dist:
                    break
                # Repeat binning:
                find = np.where((self.distance>scan_min) & (self.distance<scan_max))
                del_r_new = np.abs(scan_min - scan_max)
                N_stars = len(self.distance[find])

                if N_stars>=star_limit:
                    N_bin.append(N_stars)
                    distance_scan.append(scan_max)
                    del_r.append(del_r_new)

            if scan_max >= max_dist:
                break

        N_bin = np.array(N_bin)
        distance_scan = np.array(distance_scan)
        del_r = np.array(del_r)

        return (Table([N_bin, distance_scan, del_r], names=('N_bin', 'dist_scan', 'dr')))

    def number_density(self, **kwargs):
        """
        This function will estimate the number density (in Galactic spherical coordinates) for a given binning profile (using distance_binner)
        """
        # Bin the the distance by desiered parameters
        GB = galactic_binner(self.distance, self.l, self.b)
        dist_density = GB.distance_binner(**kwargs)

        # Estimate the span of Galactic coordinates
        delta_b = abs(max(self.b) - min(self.b))
        delta_l = abs(max(self.l) - min(self.l))

        # Calculate the volume element in each bin
        volume_sub = (dist_density['dist_scan'])**2 * (dist_density['dr']) * np.cos(np.deg2rad(delta_b)) *(delta_l) * (delta_b)

        return (dist_density['N_bin']/volume_sub)

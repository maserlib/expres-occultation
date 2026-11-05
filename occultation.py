# ================ IMPORTS

import numpy
from astropy.table import Table, Column
from astropy.units import Unit as u
from astropy.coordinates import SphericalRepresentation, CartesianRepresentation, spherical_to_cartesian, cartesian_to_spherical
from astropy.units import Quantity
import datetime 
from spacepy import pycdf
from io import StringIO

import json

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.cm as cm

myFmt = mdates.DateFormatter('%H:%M')

# ================ VARIABLES

### fylby_params

with open("flybys_list.json", "r") as f:
    flyby_params_raw = json.load(f)

flyby_params = {}
for key, val in flyby_params_raw.items():
    flyby_params[key] = {
        "spacecraft": val["spacecraft"],
        "moon": val["moon"],
        "start_time": datetime.datetime.fromisoformat(val["start_time"]),
        "end_time": datetime.datetime.fromisoformat(val["end_time"]),
    }

## available spacecraft
with open("spacecraft.json", "r") as f:
    missions_data = json.load(f)

available_missions = missions_data["available_missions"]

moon_radius = {
    'Io': 1821.6 * u('km'),
    'Europa': 1560.8 * u('km'),
    'Ganymede': 2634.1 * u('km'),
    'Callisto': 2410.3 * u('km'),
    'Titan': 2575.5 * u('km'),
    'Encelade': 252 * u('km')
}



files = {
    #'I00': '/Users/clouis/Documents/Article/BC/Juice/simus/Galileo/I00/expres_galileo_jupiter_0d-30r_jrm09_lossc-wid1deg_5kev_19951207_v01.cdf',
    #'G01': '/Users/clouis/Documents/Article/BC/Juice/simus/Galileo/G01/expres_galileo_jupiter_0d-30r_jrm09_lossc-wid1deg_5kev_19960627_v01.cdf',
    #'G02': '/Users/clouis/Documents/Article/BC/Juice/simus/Galileo/G02/expres_galileo_jupiter_0d-30r_jrm09_lossc-wid1deg_5kev_19960906_v01.cdf',
    #'C03': '/Users/clouis/Documents/Article/BC/Juice/simus/Galileo/C03/expres_galileo_jupiter_0d-30r_jrm09_lossc-wid1deg_5kev_19961104_v01.cdf',
    #'G07': '/Users/clouis/Documents/Article/BC/Juice/simus/Galileo/G07/expres_galileo_jupiter_0d-30r_jrm09_lossc-wid1deg_5kev_19970405_v01.cdf',
    #'G08': '/Users/clouis/Documents/Article/BC/Juice/simus/Galileo/G08/expres_galileo_jupiter_0d-30r_jrm09_lossc-wid1deg_5kev_19970507_v01.cdf',
    #'C09': '/Users/clouis/Documents/Article/BC/Juice/simus/Galileo/C09/expres_galileo_jupiter_0d-30r_jrm09_lossc-wid1deg_5kev_19970625_v01.cdf',
    #'C10': '/Users/clouis/Documents/Article/BC/Juice/simus/Galileo/C10/expres_galileo_jupiter_0d-30r_jrm09_lossc-wid1deg_5kev_19970916_v01.cdf',
    #'G28': '/Users/clouis/Documents/Article/BC/Juice/simus/Galileo/G28/expres_galileo_jupiter_0d-30r_jrm09_lossc-wid1deg_5kev_20000520_v01.cdf',
    #'G29': '/Users/clouis/Documents/Article/BC/Juice/simus/Galileo/G29/expres_galileo_jupiter_0d-30r_jrm09_lossc-wid1deg_5kev_20001228_v01.cdf',
#    'C30': 'data/expres_galileo_jupiter_0d-30r_jrm09_lossc-wid1deg_5kev_20010525_v01.cdf'
    #'C30': '/Users/clouis/Documents/Article/BC/Juice/simus/Galileo/C30/expres_galileo_jupiter_0d-30r_jrm09_lossc-wid1deg_5kev_20010525_v01.cdf'
}

ephem_uiowa = {
    'G01': {'GLL': 'ephem/Uiowa_Galileo_1996-179.txt',
           'Ganymede': 'ephem/UIowa_Ganymede_1996-179.txt'}, 
    'I00': {'GLL': 'ephem/Ephem_Galileo_19951207.txt'}
} 

ephem_horizons = {
    'G01': {'Ganymede': 'ephem/Horizons_Ganymede_1996-179.txt'}, 
}

# WGC files downloaded from https://wgc.jpl.nasa.gov:8443/webgeocalc/#StateVector
# check header and footer of CSV file for input parameters

def load_ephem_wgc(file='ephem-wgc.json'):
    import json
    with open(file) as json_file:
        data = json.load(json_file)
    return data


# NB for E15: Error in WGC: 
# CSPICE_N0066: CSPICE.spkezr: SPICE(SPKINSUFFDATA): [spkezr_c --> SPKEZR --> SPKEZ --> SPKGEO] Insufficient ephemeris data has been loaded to compute the state of -77 (GALILEO ORBITER) relative to 599 (JUPITER) at the ephemeris epoch 1998 MAY 31 18:31:03.184.

load_ephem_params = {
    'uiowa': ephem_uiowa,
    'horizons': ephem_horizons,
    'wgc':  load_ephem_wgc()}


# ================ CLASSES

class Ephem(object):
    """
    Ephem class includes methods to read ephemeris files (from UIowa server, from JPL/Horizons, or NASA/NAIF/WGC[CSV output] )
    """
    
    def __init__(self, flyby, target, source, verbose=False):
        self.target = target
        self.flyby = flyby
        self.source = source
        self.verbose = verbose
        self._check_inputs()
        self.file = self._get_file()
        self.data = self._load_ephem()
        nt = len(self.data)
        self.xyz = numpy.zeros((nt, 3))
        for cur_t in range(nt):
            # convert cartesian coordinates into spherical (with astropy.coordinates.spherical_to_cartesian)
            tmp_xyz = spherical_to_cartesian(
                r=self.data['Coords'][cur_t].distance, 
                lon=self.data['Coords'][cur_t].lon, 
                lat=self.data['Coords'][cur_t].lat
            )
            for i in range(3):
                self.xyz[cur_t, i] = tmp_xyz[i].value
            
    def _check_inputs(self):
        """
        Sanity checks based on the input parameters
        """
        
        if self.flyby in flyby_params.keys():
            if self.verbose:
                print('Flyby = {}'.format(self.flyby))
            if self.source in load_ephem_params.keys():
                if self.verbose:
                    print('Source = {}'.format(self.source))
                #if self.target in ['GLL', 'JUI','Juno','Clipper', flyby_params[self.flyby]['moon']]:
                if self.target in available_missions + [flyby_params[self.flyby]['moon']]:
                    if self.verbose:
                        print('Target = {}'.format(self.target))
                    if self.target in load_ephem_params[self.source][self.flyby].keys():
                        if self.verbose:
                            print('File = {}'.format(self._get_file()))
                    else:
                        raise ValueError('Target not available for this source')
                else:
                    raise ValueError('Wrong target input')
            else:
                raise ValueError('Wrong ephem source input')
        else:
            raise ValueError('Wrong flyby input')

        
    def _get_file(self):
        return load_ephem_params[self.source][self.flyby][self.target]

    def _load_ephem(self):
        dispatch = {
            'uiowa': self._load_ephem_uiowa,
            'horizons': self._load_ephem_horizons,
            'wgc': self._load_ephem_wgc_csv
        }
        return dispatch[self.source]()

    def _load_ephem_uiowa(self):
    
        dtype_raw = numpy.dtype([('year', numpy.str, 4), ('jday', numpy.str, 3), ('hour', numpy.str, 2), ('minute', numpy.str, 2), ('seconds', numpy.str, 6), ('LongW', numpy.float), ('Lat', numpy.float), ('MLat', numpy.float), ('LT', numpy.float), ('Rjup', numpy.float), ('Lshell', numpy.float), ('IoPhase', numpy.float)])
        ephem_tmp = numpy.loadtxt(self.file, dtype=dtype_raw, skiprows=5)

        ephem_out = Table()

        datetime_tmp = []
        for cur_ephem in ephem_tmp:
            (yy, jd, hh, mm, ss, _, _, _, _, _, _, _) = cur_ephem
            datetime_tmp.append(datetime.datetime.strptime('{}-{} {}:{}:{}000'.format(yy,jd,hh,mm,ss), '%Y-%j %H:%M:%S.%f'))
        ephem_out['datetime'] = Column(datetime_tmp, description='Date')

        lon = (ephem_tmp['LongW']) * u('deg')
        lat = ephem_tmp['Lat'] * u('deg')
        dis = ephem_tmp['Rjup'] * u('Rjup')

        ephem_out['Coords'] = SphericalRepresentation(lon=lon, lat=lat, distance=dis)

        ephem_out['MLat'] = Column(ephem_tmp['MLat'], unit='deg', description='Magnetic Latitude')
        ephem_out['LT'] = Column(ephem_tmp['LT'], unit='hour', description='Locat Time')
        ephem_out['Lshell'] = Column(ephem_tmp['Lshell'], unit=None, description='L-Shell')
        ephem_out['IoPhase'] = Column(ephem_tmp['IoPhase'], unit='deg', description='Io Phase angle')

        return ephem_out

    def _load_ephem_horizons(self):

        dtype_raw = numpy.dtype([('YYYY-Mon-DD', numpy.str, 11), ('HH:MM', numpy.str, 5), ('OLongW', numpy.float), ('OLat', numpy.float), ('SLongW', numpy.float), ('SLat', numpy.float), ('delta', numpy.float), ('deltadot', numpy.float)])
        ephem_buf = ''
        with open(self.file, 'r') as f:
            read_flag = False
            for line in f.readlines():
                if read_flag:
                    if line.strip() == '$$EOE':
                        read_flag = False
                    else:
                        ephem_buf += line
                else: 
                    if line.strip() == '$$SOE':
                        read_flag = True

        ephem_tmp = numpy.loadtxt(StringIO(ephem_buf), dtype=dtype_raw)

        ephem_out = Table()

        datetime_tmp = []
        for cur_ephem in ephem_tmp:
            (y_m_d, h_m, _, _, _, _, _, _) = cur_ephem
            datetime_tmp.append(datetime.datetime.strptime('{} {}'.format(y_m_d,h_m), '%Y-%b-%d %H:%M'))
        ephem_out['datetime'] = Column(datetime_tmp, description='Date')

        lon = (ephem_tmp['OLongW']) * u('deg')
        lat = ephem_tmp['OLat'] * u('deg')
        dis = (ephem_tmp['delta'] * u('AU')).to('Rjup')

        ephem_out['Coords'] = SphericalRepresentation(lon=lon, lat=lat, distance=dis)

        return ephem_out

    def _load_ephem_wgc_csv(self):

        import csv

        header = {}
        ephem_buf = []

        cur_section = None
        cur_sub_section = None
        with open(self.file, 'r') as csvfile:
            ephem_raw = csv.reader(csvfile, delimiter=',')
            for row in ephem_raw:
                if row[0] == 'State Vector Calculation':
                    cur_section = 'header'
                elif row[0] == 'Calculation Inputs':
                    cur_sub_section = 'Calculation Inputs'
                    header['Calculation Inputs'] = {}
                elif row[0] == 'State Vector Results':
                    cur_section = 'results'
                elif row[0] == 'Kernels Used':
                    cur_section = 'footer'
                    header['Kernels Used'] = []
                elif row[0] == 'UTC calendar date':
                    header['Columns'] = row
                elif cur_section == 'header':
                    if len(row) > 1:
                        header[cur_sub_section][row[0]] = row[1]
                elif cur_section == 'results':
                    ephem_buf.append(row)
                elif cur_section == 'footer':
                    header['Kernels Used'].append(row[0])
                else:
                    raise IOError('Wrong file format')

        ephem_tmp = numpy.array(ephem_buf)

        ephem_out = Table()
        datetime_tmp = []
        for cur_ephem in ephem_buf:
            datetime_tmp.append(datetime.datetime.strptime(cur_ephem[0][0:26], "%Y-%m-%d %H:%M:%S.%f"))

        ephem_out['datetime'] = Column(datetime_tmp, description='Date')
        # WGC provides E-Longitude, we want W-Longitude for Spherical coordinates
        ephem_out['Coords'] = SphericalRepresentation(lon=[(360-float(item))*u('deg') for item in ephem_tmp[:,1]], 
                                                      lat=[float(item)*u('deg') for item in ephem_tmp[:,2]], 
                                                      distance=[float(item)*u('km').to('Rjup') for item in ephem_tmp[:,3]])

        return ephem_out


class ExPRES(object):
    """
    ExPRES class loads ExPRES data (CDF files must be built with 'SrcPosition' optional variable)
    """
    
    def __init__(self, flyby, expres_file=None, debug=False):
        self._debug = debug
        self.flyby = flyby
        if expres_file is None:
            self.file = files[flyby]
        else:
            self.file = expres_file

        self.xyz, self.lon, self.Freq, self.Time = self._load_expres_data()

    def _load_expres_data(self):

        c = pycdf.CDF(self.file)
        Freq = c['Frequency'][:]
        Time = c['Epoch'][:]
        if self._debug: 
            print(c['SrcPosition'].shape)
        src_mask = c['SrcPosition'][:] < -1e30
        src_xyz = numpy.ma.masked_array(c['SrcPosition'][:], src_mask)
        src_lon = c['SrcLongitude'][:]
        if self._debug: 
            print(numpy.max(src_xyz))
            print(numpy.min(src_xyz))
            print(src_xyz.shape)
            print(type(src_xyz))
        c.close()

        return src_xyz, src_lon, Freq, Time
    
    def get_dims(self):
        nt, nf, ns, _ = self.xyz.shape
        return nt, nf, ns
    
    def get_freq_index(self, freq_value):
        freq_index = 0
        for cur_index in range(len(self.Freq)):
            if self.Freq[cur_index] < freq_value:
                freq_index = cur_index
        return freq_index
    
    def __repr__(self):
        return f'<ExPRES: {self.file} for {self.flyby}>'


class Body(object):
    """
    Body class is a simple wrapper class for a body (spacecraft or moon)
    """
    
    def __init__(self, flyby, body, source):
        self.flyby = flyby
        self.body = body
        self.source = source
        self.ephem = Ephem(flyby, body, source)
       
    def __repr__(self):
        return f'<Body: {self.body} (from {self.source}) during {self.flyby}>'


class Flyby(object):
    """
    Flyby includes all information and methods to analyse the flyby
    """
        
    def __init__(self, flyby, sc_ephem_source, moon_ephem_source, option=2, expres_file=None, debug=False):
        self._debug = debug
        self.flyby_name = flyby
        self.source = ExPRES(flyby, expres_file=expres_file)
        self.spacecraft = Body(flyby, flyby_params[flyby]['spacecraft'], sc_ephem_source)
        self.moon = Body(flyby, flyby_params[flyby]['moon'], moon_ephem_source)
        self.source_class = self._classify_source()
        self.intercept, self.intercept_front_moon = self._compute_intercept(option=option)
        
    def __repr__(self):
        return f'<Flyby: {self.flyby} with {self.source.file} ExPRES data>'
        
    def xyz_spacecraft_to_moon(self):
        vector = self.moon.ephem.xyz - self.spacecraft.ephem.xyz
        if self._debug:
            print(vector.shape)
        return vector
    
    def check_flyby_distance(self):
        fig, ax = plt.subplots()
        ax.xaxis.set_major_formatter(myFmt)
        

        ax.plot(
            self.moon.ephem.data['datetime'], 
            numpy.sqrt(numpy.sum(self.xyz_spacecraft_to_moon()**2, axis=1))*71492/moon_radius[flyby_params[self.flyby_name]['moon']])
        ax.set_xlabel("Time of {}".format(self.moon.ephem.data['datetime'][0].date()))
        ax.set_ylabel("Distance in Moon Radius")
        ax.set_title("SC to Moon Distance")
        plt.show()
        
        if self._debug:
            print(numpy.min(numpy.sqrt(numpy.sum(self.xyz_spacecraft_to_moon()**2, axis=1))*71492/2634))

    def xyz_spacecraft_to_source(self):

        nt, nf, ns = self.source.get_dims()

        gll_xyz = numpy.ma.masked_array(numpy.empty_like(self.source.xyz), self.source.xyz.mask)
        xyz_gll_src = numpy.ma.masked_array(numpy.empty_like(self.source.xyz), self.source.xyz.mask)

        for cur_t in range(nt):
            tmp_gll_xyz = numpy.tile(self.spacecraft.ephem.xyz[cur_t, :], (nf, ns)).reshape((nf, ns, 3))
            gll_xyz[cur_t, :, :, :] = tmp_gll_xyz

        if self._debug:
            print(self.source.xyz.shape, gll_xyz.shape)    
            print(type(self.source.xyz), type(gll_xyz))    
            print(xyz_gll_src.shape)
            print(type(xyz_gll_src))    

        xyz_gll_src = self.source.xyz - gll_xyz

        if self._debug:
            print(type(xyz_gll_src))

        return gll_xyz, xyz_gll_src


    def check_src_distance(self, freq=4):
        cur_f = self.source.get_freq_index(freq)
        if self._debug:
            print(self.source.Freq[cur_f])

        fig, ax = plt.subplots()
        ax.xaxis.set_major_formatter(myFmt)

        _, xyz_gll_src = self.xyz_spacecraft_to_source()
        #for cur_f in range(nf):

        

        for cur_s in range(xyz_gll_src.shape[2]):
                ax.plot(self.source.Time, numpy.sqrt(numpy.sum(xyz_gll_src[:,cur_f,cur_s,:]**2, axis=1)), color = 'blue')

        ax.set_xlabel("Time of {}".format(self.source.Time[0].date()))
        ax.set_ylabel("Distance in Moon Radius")
        ax.set_title("SC to Radio Sources Distance (@{:5.2f} MHz)".format(self.source.Freq[cur_f]))
        ax.set_ylim((0,20))
        plt.show()

    def _compute_intercept(self, option=2):
        # NB: only option 1 has been tested.

        nt, nf, ns = self.source.get_dims()

        intercept = numpy.zeros((nt, nf, ns))
        intercept_front_moon = numpy.zeros((nt, nf, ns))
        _, xyz_gll_src = self.xyz_spacecraft_to_source()
        xyz_gll_moon = self.xyz_spacecraft_to_moon()
        if self._debug:
            print(xyz_gll_src.shape)
            print(xyz_gll_moon.shape)

        # compute interception distance:
        # [option 1] => interception = || (GLL-GAN vector) X (GLL-SRC vector) || / || (GLL-SRC vector) || 
        # [option 2] => interception = || ((GLL-SRC vector).(GLL-GAN vector) * (GLL-SRC vector) / || (GLL-SRC vector) ||) - (GLL-GAN vector)||

        # ================== [option 1] ==================
        if option == 1: 
            for cur_t in range(nt):
                tmp_xyz_gll_moon = numpy.tile(xyz_gll_moon[cur_t, :], (nf, ns)).reshape((nf*ns, 3))
                if self._debug:
                    print(tmp_xyz_gll_moon.shape)
                tmp_xyz_gll_src = xyz_gll_src[cur_t, :, :, :].reshape(nf*ns, 3)
                if self._debug:
                    print(tmp_xyz_gll_src.shape)
                tmp_cross = numpy.cross(tmp_xyz_gll_moon, tmp_xyz_gll_src)
                if self._debug:
                    print(tmp_cross.shape)
                tmp_cross_norm = numpy.sqrt(numpy.sum(tmp_cross**2, axis=1))
                if self._debug:
                    print(tmp_cross.shape)
                tmp_xyz_gll_src_norm = numpy.sqrt(numpy.sum(tmp_xyz_gll_src**2, axis=1))
                if self._debug:
                    print(tmp_xyz_gll_src_norm.shape)   
                tmp_intercept = tmp_cross_norm/tmp_xyz_gll_src_norm
                if self._debug:
                    print(tmp_intercept.shape)   
                intercept[cur_t, :, :] = tmp_intercept.reshape((nf, ns))
        

        # ================== [option 2] ==================

        if option == 2:
            for cur_t in range(nt):
                # Repeat moon position for all frequencies and sources
                tmp_xyz_gll_moon = numpy.tile(xyz_gll_moon[cur_t, :], (nf*ns, 1))
                tmp_xyz_gll_src = xyz_gll_src[cur_t, :, :, :].reshape(nf*ns, 3)

                # Vector from spacecraft to moon and spacecraft to source
                vec_sc_moon = tmp_xyz_gll_moon
                vec_sc_src = tmp_xyz_gll_src

                # Perpendicular distance
                tmp_cross = numpy.cross(vec_sc_moon, vec_sc_src)
                tmp_cross_norm = numpy.linalg.norm(tmp_cross, axis=1)
                tmp_xyz_gll_src_norm = numpy.linalg.norm(vec_sc_src, axis=1)
                tmp_intercept = tmp_cross_norm / tmp_xyz_gll_src_norm
                tmp_intercept_infrontofmoon = tmp_cross_norm / tmp_xyz_gll_src_norm

                # Dot product to check if moon is in front of spacecraft along the source direction
                dot = numpy.sum(vec_sc_moon * vec_sc_src, axis=1)
                tmp_intercept[dot <= 0] = numpy.inf  # set to large value if moon is behind spacecraft
                tmp_intercept_infrontofmoon[dot>=0] = numpy.inf  # True if moon is in front
                
                # Reshape back
                intercept[cur_t, :, :] = tmp_intercept.reshape((nf, ns))
                intercept_front_moon[cur_t, :, :] = tmp_intercept_infrontofmoon.reshape((nf, ns))
                

        print(intercept.shape)
        intercept_mask = intercept > 1e20
        intercept_front_moon_mask = intercept_front_moon > 1e20

        intercept = numpy.ma.masked_array(intercept, intercept_mask)
        intercept_front_moon = numpy.ma.masked_array(intercept_front_moon, intercept_front_moon_mask)
        print(numpy.min(intercept))
        print(numpy.max(intercept))
        return intercept, intercept_front_moon

    
    def plot_intercept(self, src_class=None, freq=4,zoomin=False,radius_occultation=1.,savefig=False, option = 1):
        cur_f = self.source.get_freq_index(freq)
        ns = self.source.get_dims()[2]
        fig, ax = plt.subplots()
        ax.xaxis.set_major_formatter(myFmt)

        intercept = self.intercept.copy()
        intercept_front_mask =  self.intercept_front_moon.copy()

        title_class_tag = ''
        if src_class is not None:
            intercept.mask = (self.source_class != src_class)
            intercept_front_mask.mask = (self.source_class != src_class)
            title_class_tag = ' ("{}"-source)'.format(src_class)
        else:
            title_class_tag = ' (all sources)'
        for cur_s in range(ns):
            label_SMS = 'Source-moon-spacecraft' if cur_s == 0 else None
            label_SSM = 'Source-spacecraft-moon' if cur_s == 0 else None
            if option == 1:
                ax.plot(self.source.Time,intercept[:,cur_f,cur_s]*71492/moon_radius[flyby_params[self.flyby_name]['moon']], color = "blue")
            if option == 2:
                ax.plot(self.source.Time,intercept[:,cur_f,cur_s]*71492/moon_radius[flyby_params[self.flyby_name]['moon']], color = "blue", label=label_SMS)
                ax.plot(self.source.Time,intercept_front_mask[:,cur_f,cur_s]*71492/moon_radius[flyby_params[self.flyby_name]['moon']], color = "grey", label=label_SSM)
                
        ax.legend()
        ax.plot([self.source.Time[0], self.source.Time[-1]], [radius_occultation, radius_occultation], '--')
        if zoomin == True:
            ax.set_ylim([0,4])
        ax.set_xlabel("Time of {}".format(self.source.Time[0].date()))
        ax.set_ylabel("Distance in Moon Radius")
        ax.set_title("Intercept Distance @{:5.2f} MHz {}".format(self.source.Freq[cur_f], title_class_tag))
        if savefig == True:
            if src_class is not None:
                plt.savefig('figures/Intercept_distance_source'+src_class, dpi=300, bbox_inches='tight')
            else:
                plt.savefig('figures/Intercept_distance_allsources', dpi=300, bbox_inches='tight')
        plt.show()
        
    def check_trajectory(self, plane='xy', freq=4): #xyz_gll, xyz_gan, xyz_src,):

        cur_f = self.source.get_freq_index(freq)

        cur_axis = {'x': 0, 'y': 1, 'z': 2}
        cur_x = cur_axis[plane[0]]
        cur_y = cur_axis[plane[1]]

        ns = self.source.get_dims()[2]
        plt.figure(figsize=(10,10))
        plt.plot(self.spacecraft.ephem.xyz[:,cur_x], self.spacecraft.ephem.xyz[:,cur_y])
        plt.plot([self.spacecraft.ephem.xyz[0,cur_x]], self.spacecraft.ephem.xyz[0,cur_y], '+')
        plt.plot(self.moon.ephem.xyz[:,cur_x], self.moon.ephem.xyz[:,cur_y])

        for cur_s in range(ns):
            plt.plot(self.source.xyz[:,cur_f,cur_s,cur_x], self.source.xyz[:,cur_f,cur_s,cur_y], '+')

        plt.xlim((-16,16))
        plt.ylim((-16,16))
        plt.xlabel('{} (Rj)'.format(plane[0]))
        plt.ylabel('{} (Rj)'.format(plane[1]))
        
    def check_src_position_1D(self, axis='x', freq=4):

        cur_f = self.source.get_freq_index(freq)

        cur_axis = {'x': 0, 'y': 1, 'z': 2}
        cur_x = cur_axis[axis]

        ns = self.source.get_dims()[2]
        plt.figure(figsize=(10,10))

        for cur_s in range(ns):
            plt.plot(self.source.Time, self.source.xyz[:,cur_f,cur_s,cur_x], '+')

        plt.ylim((-5,5))
        plt.ylabel('{} (Rj)'.format(axis))
    
    def check_src_position_2D(self, plane='xy', freq=4):

        cur_f = self.source.get_freq_index(freq)

        cur_axis = {'x': 0, 'y': 1, 'z': 2}
        cur_x = cur_axis[plane[0]]
        cur_y = cur_axis[plane[1]]

        ns = self.source.get_dims()[2]
        plt.figure(figsize=(10,10))

        for cur_s in range(ns):
            print(numpy.unique(xyz_src[:,cur_f,cur_s,cur_x]))
            plt.plot(self.source.xyz[:,cur_f,cur_s,cur_x], self.source.xyz[:,cur_f,cur_s,cur_y], '+')

        plt.xlim((-5,5))
        plt.ylim((-5,5))
        plt.xlabel('{} (Rj)'.format(plane[0]))
        plt.ylabel('{} (Rj)'.format(plane[1]))
        
    def _classify_source(self):
        nt, nf, ns = self.source.get_dims()
        source_class = numpy.empty((nt, nf, ns), dtype=numpy.dtype('U1'))
        for cur_t in range(nt):
            tmp_sc_lon = self.spacecraft.ephem.data['Coords'][cur_t].lon
            tmp_src_z = self.source.xyz[cur_t, :, :, 2].reshape(nf*ns)
            tmp_src_lon = numpy.tile(self.source.lon[cur_t, :], nf).reshape(nf*ns) * u('deg')
            tmp_dlon = (((tmp_src_lon - tmp_sc_lon).value + 180) % 360) - 180 
            tmp_class = numpy.empty((nf*ns), dtype=numpy.dtype('U1'))
            tmp_class[(tmp_src_z > 0) & (tmp_dlon > 0)] = 'A'
            tmp_class[(tmp_src_z > 0) & (tmp_dlon < 0)] = 'B'
            tmp_class[(tmp_src_z < 0) & (tmp_dlon > 0)] = 'C'
            tmp_class[(tmp_src_z < 0) & (tmp_dlon < 0)] = 'D'
            source_class[cur_t, :, :] = tmp_class.reshape((nf, ns))
        source_class[self.source.xyz[:,:,:,0].mask] = ''
        return source_class    

# ================ FUNCTIONS

def check_ephem(flyby, target, sources=None):

    if sources is None:
        sources = list(load_ephem_params.keys())

    ephem = {}
    for source in sources:
        ephem[source] = Ephem(flyby, target, source).data
        print(f'{target}@{source}: {len(ephem[source])}')

    plt.figure(figsize=(10,5))
    plt.title(target)
    for source in sources:
        plt.plot(ephem[source]['datetime'], [item.lon.deg for item in ephem[source]['Coords']], label=source)
    plt.legend()
    plt.ylabel('Longitude W (deg)')
    plt.show()

    plt.figure(figsize=(10,5))
    plt.title(target)
    for source in sources:
        plt.plot(ephem[source]['datetime'], [item.lat.deg for item in ephem[source]['Coords']], label=source)
    plt.legend()
    plt.ylabel('Latitude (deg)')
    plt.show()

    if len(sources) > 1:
        plt.figure(figsize=(10,5))
        plt.title(target)
        for source in sources[1:]:
            plt.plot(ephem[source]['datetime'], [item0.lon.deg - item.lon.deg for item0, item in zip(ephem[sources[0]]['Coords'], ephem[source]['Coords'])], label='{}-{}'.format(sources[0], source))
        plt.legend()
        plt.ylabel('delta Longitude W (deg)')
        plt.show()

        plt.figure(figsize=(10,5))
        plt.title(target)
        for source in sources[1:]:
            plt.plot(ephem[source]['datetime'], [item0.lat.deg - item.lat.deg for item0, item in zip(ephem[sources[0]]['Coords'], ephem[source]['Coords'])], label='{}-{}'.format(sources[0], source))
        plt.legend()
        plt.ylabel('delta Latitude (deg)')
        plt.show()


def get_flyby_table():

    col0 = list(flyby_params.keys())
    col1 = []
    col2 = []
    col3 = []
    col4 = []

    for cur_flyby in col0:
        col1.append(flyby_params[cur_flyby]['spacecraft'])
        col2.append(flyby_params[cur_flyby]['moon'])
        col3.append(flyby_params[cur_flyby]['start_time'])
        col4.append(flyby_params[cur_flyby]['end_time'])
    
    table = Table((
        Column(col0, name='Flyby'),
        Column(col1, name='Spacecraft'), 
        Column(col2, name='Moon'), 
        Column(col3, name='Start Time'), 
        Column(col4, name='End Time')
    ))
    table.add_index('Flyby')
    
    return table

flyby_table = get_flyby_table()

# Function to load GLL/PWS data from Das2 service at UIowa

def load_data_from_das2(flyby_id):
    import das2 
    row_id = flyby_table.loc_indices[flyby_id]
    
    d2src = das2.get_source('site:/uiowa/galileo/pws/survey_electric/das2')
#    d2src = das2.get_source('tag:das2.org,2012:site:/uiowa/galileo/pws/survey_electric')
    print(d2src)
    d2dataset = d2src.get({'time': (flyby_params[flyby_id]['start_time'].isoformat(), 
                                    flyby_params[flyby_id]['end_time'].isoformat(), 4.0)})[0] 
    
    d2a_t = d2dataset['time']['center'].array
    d2a_f = d2dataset['frequency']['center'].array
    d2a_sp = d2dataset['electric']['center'].array
    return d2a_t, d2a_f, d2a_sp


# Function to display the the modelled occultations (with observed data for GLL)
def full_plots(flyby_id, expres_file=None, debug=False,radius_occultation=1.,savefig=False, figsize = (10, 7), option = 2, frequency_to_look_at = 10):
    """Function display occultation data, including observed data for GLL flybys.
    
    This function requires the `occultation.py` scripts, the `ephem-wgc.json` file and the corresponding ephemeris
    (state vector) CSV files (exported from WebGeocalc), as named in `ephem-wgc.json`.
    
    :param flyby_id: Id of the flyby
    :param expres_file: CDF file path of ExPRES results
    
    :type flyby_id: str
    :type flyby_id: str or Path
    """
    
    # find current flyby info
    cur_flyby = flyby_table[flyby_table.loc_indices[flyby_id]]
 
    # Load and display observed data if GLL
    if cur_flyby['Spacecraft'] == 'GLL':
        d2a_t, d2a_f, d2a_sp = load_data_from_das2(flyby_id)
        import matplotlib.colors as colors 

        print(numpy.min(d2a_sp))
        print(numpy.max(d2a_sp))

        plt.figure(figsize=(20, 13))
        scaleZ = colors.LogNorm(vmin=numpy.min(d2a_sp), vmax=numpy.max(d2a_sp)) 
        plt.pcolormesh(d2a_t, d2a_f, d2a_sp, cmap='jet', vmin=1e-16, vmax=1e-12, norm=scaleZ) 
        plt.yscale("log")
        plt.ylim((1e5,5.7e6))
        plt.xlabel('Time')
        plt.ylabel('Frequency (Hz)')
        plt.show()

    # define a Flyby object with the input parameters (here, default is to use 'wgc' ephemeris)
    f = Flyby(flyby_id, 'wgc', 'wgc', expres_file=expres_file, option = option)
    
    # check if moon radius is present for the considered moon:
    if flyby_params[flyby_id]['moon'] not in moon_radius:
        moon_radius_new = float(input(f"Moon radius is not defined for this moon. Please enter the {cur_flyby['Moon']} radius (in km)"))
        moon_radius[f"{cur_flyby['Moon']}"] = moon_radius_new * u('km')
    cur_moon_radius  = moon_radius[flyby_params[flyby_id]['moon']].value
    # compute and check various distances  
    print('this is the distance to the flyby target')
    f.check_flyby_distance()
    print('this is the distance to the radio sources')
    f.check_src_distance(freq = frequency_to_look_at) 
    print('this is the intercept distance at the default \n If a line goes below 1 \n if the line is blue there should be an occultation \n if the line is grey the spacecraft is located in-between the sources and the moon')
    f.plot_intercept(src_class='A',zoomin=True,radius_occultation=radius_occultation,savefig=True, option = option, freq = frequency_to_look_at)
    f.plot_intercept(src_class='B',zoomin=True,radius_occultation=radius_occultation,savefig=True, option = option, freq = frequency_to_look_at)
    f.plot_intercept(src_class='C',zoomin=True,radius_occultation=radius_occultation,savefig=True, option = option, freq = frequency_to_look_at)
    f.plot_intercept(src_class='D',zoomin=True,radius_occultation=radius_occultation,savefig=True, option = option, freq = frequency_to_look_at)
    f.plot_intercept(zoomin=True,radius_occultation=radius_occultation,savefig=True, option = option, freq = frequency_to_look_at)
    
    
    if debug:
        print(f'f.intercept shape: {f.intercept.shape}')
    if debug:
        print(f'f.source_class shape: {f.source_class.shape}')
        
    nt, nf, ns = f.intercept.shape
    print(nt*nf)
    print(numpy.any((f.intercept*71492/1821.6 < 1) & (f.source_class == 'A'), axis=2).shape)

    occult_radius = radius_occultation  # change here if you want to have occultation including a larger object (set to 1.1 for 10% larger)    
    print(f'The occultation radius is set to {occult_radius} (1 = no atmosphere)')
    


    
    src_class_list = ['A', 'B', 'C', 'D']
    src_all = {}
    src_occult = {}
    for cur_class in src_class_list:
        # create a mask for the current source class
        cur_class_mask = numpy.logical_not(numpy.any((f.source_class == cur_class), axis=2))
        # create a masked array of ones (with (nt, nf) dimensions) for the current class (all sources)
        src_all[cur_class] = numpy.ma.masked_array(numpy.ones((nt, nf)), cur_class_mask)
        # create a mask for intercepted sources of current class
        cur_intercept_mask = numpy.any((f.intercept*71492/cur_moon_radius < occult_radius) & (f.source_class == cur_class), axis=2)
        # joining the inverted intercepted sources and inverted current source class masks
        cur_occult_mask = numpy.logical_not(cur_intercept_mask) & numpy.logical_not(cur_class_mask)
        # create a masked array of ones (with (nt, nf) dimensions) for visible sources fo the current class
        src_occult[cur_class] = numpy.ma.masked_array(numpy.ones((nt, nf)), numpy.logical_not(cur_occult_mask))
        #src_occult[cur_class] = numpy.ma.masked_array(numpy.ones((nt, nf)), cur_occult_mask) # inverting value compared to line above for plots to come
    

    print(numpy.min(src_occult['A']), numpy.sum(src_occult['A']))
    print(numpy.max(src_occult['B']), numpy.sum(src_occult['B']))
    
    # print the visible sources for each source class (1st the visible radio sources, then including occultation)

    # Define a two-color colormap: 0 = light gray, 1 = purple
    cmap_occultation = mcolors.ListedColormap(["#FFFFFF", "#440154"])
    
    #cmap_occultation.set_bad(color="#C8C8C8")
    bounds = [0, 0.5, 1]  # boundaries for discrete mapping
    norm = mcolors.BoundaryNorm(bounds, cmap_occultation.N)
    


    for cur_class in src_class_list:
        plt.figure(figsize=figsize)
        plt.pcolormesh(f.source.Time, f.source.Freq*1e6, src_all[cur_class].transpose(),
                             zorder=2, alpha=0.5,
                             cmap=cmap_occultation, norm=norm)
        plt.yscale("log")

        plt.xlabel("Time of {}".format(f.source.Time[0].strftime("%d %B %Y")),fontsize=14)
        plt.ylabel('Frequency (Hz)',fontsize=14)
        plt.title('{} sources'.format(cur_class),fontsize=16)

        cbar = plt.colorbar(ticks=[0.25, 0.75])
        cbar.ax.set_yticklabels(["No visible sources", "Visible sources"])
        
        plt.gcf().axes[0].xaxis.set_major_formatter(myFmt)
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gca().xaxis.set_minor_locator(mdates.AutoDateLocator())

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=figsize)
        plt.pcolormesh(f.source.Time, f.source.Freq*1e6, src_occult[cur_class].transpose(),
                             zorder=2, alpha=0.5,
                             cmap=cmap_occultation, norm=norm)
        plt.yscale("log")
        plt.xlabel("Time of {}".format(f.source.Time[0].strftime("%d %B %Y")),fontsize=14)
        plt.ylabel('Frequency (Hz)',fontsize=14)
        plt.title('{} sources'.format(cur_class),fontsize=16)
    
        
        plt.gcf().axes[0].xaxis.set_major_formatter(myFmt)
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())  
        plt.gca().xaxis.set_minor_locator(mdates.AutoDateLocator())

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        cbar = plt.colorbar(ticks=[0.25, 0.75])
        cbar.ax.set_yticklabels(["No visible sources", "Visible sources"])

        plt.tight_layout()
        if savefig == True:
            plt.savefig('figures/Moon_flyby_occultation_simu_source'+cur_class+'.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    plt.figure(figsize=figsize)
    for cur_class in src_class_list:
        plt.pcolormesh(f.source.Time, f.source.Freq*1e6, src_occult[cur_class].transpose(),
                              zorder=2, alpha=0.3,
                             cmap=cmap_occultation, norm=norm)
        plt.yscale("log")

    plt.ylim(1e5, 5.6e6)
    plt.xlabel("Time of {}".format(f.source.Time[0].strftime("%d %B %Y")),fontsize=14)
    plt.ylabel('Frequency (Hz)',fontsize=14)
    plt.title('All sources',fontsize=16)
    
    plt.gcf().axes[0].xaxis.set_major_formatter(myFmt)
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())  
    plt.gca().xaxis.set_minor_locator(mdates.AutoDateLocator())
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    cbar = plt.colorbar(ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(["No visible sources", "Visible sources"])

    plt.tight_layout()
    if savefig == True:
        plt.savefig('figures/Moon_flyby_occultation_simu_allsource.png', dpi=300, bbox_inches='tight')
    plt.show()
            
    plt.figure(figsize=figsize)
    
    # prepare the final plot 
    if cur_flyby['Spacecraft'] == 'GLL':
        plt.pcolormesh(d2a_t, d2a_f, d2a_sp, cmap='jet', vmin=1e-16, vmax=1e-12, norm=scaleZ)
        xlim = (d2a_t[0][0], d2a_t[-1][0])
        ylim = (1e5,5.7e6)
    else:
        xlim = (cur_flyby['Start Time'], cur_flyby['End Time'])
        ylim = (80e3,45e6)
   
    # defining colors for each source class
    class_color = {'A': '#F0F0F0', 'B': '#C0C0C0', 'C': '#909090', 'D': '#606060'}
    classes = list(class_color.keys())
    colors = list(class_color.values())
    # and plot them
    for cur_class in src_class_list:
        plt.contour(f.source.Time, f.source.Freq*1e6, src_occult[cur_class].mask.transpose()*1, [0.5], linewidths=[4], colors=([class_color[cur_class]]))

    # Add colorbar to figure
 
    colors = list(class_color.values())
    classes = list(class_color.keys())
    cmap_class = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(range(len(classes)+1), cmap_class.N)

    # Create a fake mappable for the colorbar
    sm = cm.ScalarMappable(cmap=cmap_class, norm=norm)
    sm.set_array([])  # required
    cbar = plt.colorbar(sm, ticks=[0.5, 1.5, 2.5, 3.5])
    cbar.set_ticklabels(classes)
    cbar.set_label("Class", fontsize=14)
    cbar.ax.tick_params(labelsize=14)


    # finalize plot
    print(xlim)
    plt.yscale("log")
    plt.ylim(ylim)
    plt.xlabel("Time of {}".format(f.source.Time[0].strftime("%d %B %Y")),fontsize=14)
    plt.xlim(xlim)
    plt.plot(xlim, (10e6, 10e6), '--')
    plt.ylabel('Frequency (Hz)',fontsize=14)
    plt.title('Contours of Auroral-Oval (M-shell=30) Radio Source Occultation during {} flyby'.format(flyby_id),fontsize=16)
    plt.gcf().axes[0].xaxis.set_major_formatter(myFmt)
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator()) 
    plt.gca().xaxis.set_minor_locator(mdates.AutoDateLocator())
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    if savefig == True:
        plt.savefig('figures/Moon_flyby_occultation_simu.png', dpi=300, bbox_inches='tight')
    plt.show()

    

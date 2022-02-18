import numpy as np
import pandas as pd
import os, gc, warnings, datetime, json, re

from scipy.optimize import leastsq, curve_fit
from datetime import datetime
from scipy import signal
from scipy.special import wofz
from astropy.io import fits
from os import sys

class Calc:
    '''
    A collection of calculations used to work with HST/COS data
    '''

    def GetData(self, param, datadirs):

        # Read through the contents of datadirs.
        dir_contents    = os.listdir(datadirs)
        
        if param["filenames"]["split_files"] == "no":
            fits_files = sorted([fn for fn in dir_contents if fn.startswith(param["filenames"]["filename_start"])\
                         and fn.endswith(param["filenames"]["filename_end"])])
        else:
            fits_files	= sorted([fn for fn in dir_contents if fn.endswith('sum1.fits')\
                	  or fn.endswith('sum2.fits') or fn.endswith('sum3.fits') or fn.endswith('sum4.fits')])

        # Remove the last sum file as it is faulty in some way
        #fits_files  = fits_files[:-1]                          
                                  
        # Read the HEADER of the fits file.
        for i in range(len(fits_files)):
            hdul        = fits.open(datadirs+fits_files[i])
            for hdu in hdul:
                tbheader0 = hdul[0].header.copy()
                tbheader1 = hdul[1].header.copy()
            hdul.close()
            gc.collect()
            
            try:
                print("\tAVM shift:",str(tbheader0['POSTARG1'])+"\"","\tEXP:",str(round(float(tbheader1['EXPTIME'])))+"s,",\
                "\tDate:",tbheader1['DATE-OBS']+",","Time:",tbheader1['TIME-OBS'],"UTC")
            except KeyError:
                # Remove the last sum file as it is faulty in some way
                fits_files  = fits_files[:-1] 
                pass

        # Initialise a 2D array.
        # The range(4) comes from the data format used: 0 = wavelength, 1 = flux, 2 = flux_err, 3 = obs_time
        d = [[[] for i in range(4)] for j in range(len(fits_files))]
        
        # Determine if the data was taken before 2017
        # when a detector shift was introduced.
        date_split = re.findall('[0-9]+', datadirs)
        if int(date_split[0]) < 2017:
            epoch = 'pre'
        else:
            epoch = 'post'
        
        part = param["BetaPictoris"]["part"]

        for i in range(len(fits_files)):
            # Extract the data from the fits files.
            d[i][0],d[i][1],d[i][2],d[i][3] = self.ExtractData(datadirs+fits_files[i],part,epoch)
            # To avoid dividing my 0 when calculating the ratios below, NaN and 0's are
            # replaced by a small number. The uncertainty values are replaced by 1 (extremely uncertain).
            d[i][1] = self.ReplaceWithSmallNumber(d[i][1])
            d[i][2] = self.ReplaceWithOne(d[i][2])

        # We sort the data ordered by start of exposure
        d = sorted(d,key=lambda l:l[-1])

        return d

    def ExtractData(self, fits_file, part, epoch):
        ''' Extracts the header information and
        data from fits files'''
        
        # Open the selected fits file.
        hdul        = fits.open(fits_file)
        
        # Read the data and header from the fits file.
        for hdu in hdul:
            tbdata      = hdul[1].data.copy()
            tbheader 	= hdul[1].header.copy()
        
        # To avoid too many open files we close hdul.
        hdul.close()
        gc.collect()

        net         = tbdata['NET']
        gcounts     = tbdata['GCOUNTS']
        exptime     = np.array([tbdata['EXPTIME']])
        w           = tbdata['WAVELENGTH']
        f           = tbdata['FLUX'] 
        a           = net*exptime.T
        
        t_obs         = datetime.strptime(tbheader['DATE-OBS']+" "+tbheader['TIME-OBS'], '%Y-%m-%d %H:%M:%S')
        
        # Make sure no array element is 0
        a = self.ReplaceWithSmallNumber(a)
        
        # Uncertainy
        e   = np.sqrt(gcounts+1)*(f / a)
        #e   = self.ReplaceWithSmallNumber(e)
        e   = self.ReplaceWithOne(e)
        
        # DATA matrix is initialised with data
        # depeding on the detector of interest.
        if part == 'B':
            DATA = [w[1],f[1],e[1]]
        if part == 'A':
            DATA = [w[0],f[0],e[0]]

        # We shift the data so that they all match
        # adding the difference either before or after the spectra
        units   = 3864
        w, f, e = self.ShiftDATA(DATA,units,part,epoch)        
        
        return w, f, e, t_obs        


    def ReplaceWithSmallNumber(self, X):
        X[np.isnan(X)] = 0
        X[X == 0] = 1e-20
        return X


    def ReplaceWithOne(self, X):
        X[np.isnan(X)] = 0
        X[X == 0] = 1.0
        return X


    def RegionSelect(self, wave, lambda1, lambda2):
        for i in range(len(wave)):
            if wave[i] > lambda1 and wave[i] < lambda1+(wave[i]-wave[i-1]):
                start = i
            if wave[i] > lambda2 and wave[i] < lambda2+(wave[i]-wave[i-1]):
                stop = i
        return start, stop


    def ShiftSpec(self, param, S):
        # This routine correlates the spectrum: spec
        # with the reference spectrum: ref
        
        # Select wavelength of the chosen line
        chosen_line = param["lines"]["chosen_line"]
        Line        = param["lines"]["line"][chosen_line]["Wavelength"]
        c_light     = 299792458

        # The regions used for the cross-correlation will depend on what region you are looking at.
        # The B region is towards the blue and the A region is towards the red.
        if param["BetaPictoris"]["part"] == 'A':    
            lambda1 = param["regions"]["align"]["A1"]
            lambda2 = param["regions"]["align"]["A2"]
        else:
            lambda1 = param["regions"]["align"]["B1"]
            lambda2 = param["regions"]["align"]["B2"]

        
        
        # We create an empty nested array which will be filled with data
        Ds  = [ [[[] for i in range(4)] for j in range(len(S[k]))] for k in range(len(S)) ]

        ref_wave    = S[0][0][0]    # Wavelength of spec to be shifted
        ref_spec    = S[0][0][1]    # Reference spectrum

        # We select the relevant region. 
        start, stop = self.RegionSelect(S[0][0][0], lambda1, lambda2)

        # The spectra used for the correlation is cut so that it contains
        # the specified region.
        ref_wave_c  = S[0][0][0][start:stop]    # Wavelength of spec to be shifted
        ref_spec_c  = S[0][0][1][start:stop]    # Reference spectrum

        ref_fit         = np.polyfit(ref_wave_c, ref_spec_c, 1)
        ref_spec_c_n    = ref_spec_c/np.polyval(ref_fit,ref_wave_c) - 1.

        for i in range(len(S)):
            print('\n\n')
            if i == 0:
                print("\033[34mShifting the "+str(i+1)+"st observation:\033[0m")
            if i == 1:
                print("\033[34mShifting the "+str(i+1)+"nd observation:\033[0m")
            if i > 1:
                print("\033[34mShifting the "+str(i+1)+"th observation:\033[0m")


            for j in range(len(S[i])):
                print('\n\nObservations done at:',S[i][j][3])

                spec        = S[i][j][1]    # Spectrum to be shifted
                error       = S[i][j][2]    # Error on spec to be shifted

                spec_c      = S[i][j][1][start:stop]    # Spectrum to be shifted

                spec_fit        = np.polyfit(ref_wave_c, spec_c, 1)
                
                spec_c_n        = spec_c/np.polyval(spec_fit,ref_wave_c) - 1.

                c           = np.correlate(spec_c_n,ref_spec_c_n,mode='full')
                x           = np.arange(c.size)
                c_max       = np.argmax(c)          # Maximum correlation
               
                print("_"*46)
                if ref_spec_c_n.size-1 > c_max:        # If spectrum is redshifted
                    #ref_c.size-1 because python starts index at 0
                    shift         = ref_wave_c[ref_spec_c_n.size-1]-ref_wave_c[np.argmax(c)]
                    units         = (ref_spec_c_n.size-1)-np.argmax(c)
                    RV_shift      = ((shift/Line)*c_light)/1.e3
                    
                    if abs(units) < 100:
                        print("\n Pixel shift [pix]:\t",units)
                        print(" W shift [A]:\t\t",round(shift,2))
                        print(" RV shift [km/s]:\t",round(RV_shift))          
                    else:
                        print("Big shift, likely airglow observation")

                    zeros         = np.zeros(units)
                    if units     != 0:
                        spec        = np.concatenate((zeros, spec), axis=0)[:-units]
                        error       = np.concatenate((zeros, error), axis=0)[:-units]
                else:                           # If spectrum is blueshifted
                    c             = np.correlate(ref_spec_c_n,spec_c_n,mode='full')
                    shift         = ref_wave_c[np.argmax(c)]-ref_wave_c[ref_spec_c_n.size-1]
                    units         = np.argmax(c)-(ref_spec_c_n.size-1)
                    units_pos     = abs(units)
                    RV_shift      = ((shift/Line)*c_light)/1.e3

                    if units_pos < 100:
                        print("\n Pixel shift [pix]:\t",units)
                        print(" W shift [A]:\t\t",round(shift,2))
                        print(" RV shift [km/s]:\t",round(RV_shift)) 
                    else:
                        print("Big shift, likely airglow observation")
                    zeros         = np.zeros(units_pos)
                    if units     != 0:
                        spec        = np.concatenate((spec, zeros), axis=0)[units_pos:]
                        error       = np.concatenate((error, zeros), axis=0)[units_pos:]
                print("_"*46)

                Ds[i][j][0],Ds[i][j][1],Ds[i][j][2],Ds[i][j][3] = ref_wave, spec, error, S[i][j][3]
            
        return np.array(Ds)
        

    def NormSpec(self, param, S):
        ''' Calculate the factor to multiply the spectra by to match the flux level of the first spectrum.
        The factor is calculated using the region [n1:n2].
        
        S = all the spectra
        
        '''
        print("\n\nNormalising the spectra relative to first spectrum...")
        
        # Check what part of the spectrum we are to normalise
        if param["BetaPictoris"]["part"] == 'A':    
            n1 = param["regions"]["norm"]["A1"]
            n2 = param["regions"]["norm"]["A2"]
        else:
            n1 = param["regions"]["norm"]["B1"]
            n2 = param["regions"]["norm"]["B2"]

        start, stop = self.RegionSelect(S[0][0][0], n1, n2)

        R   = [ [[] for j in range(len(S[k]))] for k in range(len(S)) ]
        Dn = [ [[[] for i in range(4)] for j in range(len(S[k]))] for k in range(len(S)) ]  

        obs = []
        for i in sorted(param["datadirs"]):
            obs.append(i)

        for i in range(len(S)):
            for j in range(len(S[i])):
                S[i][j][2][start:stop] = self.ReplaceWithSmallNumber(S[i][j][2][start:stop])

                R[i][j] = np.average(S[0][0][1][start:stop]) / np.average(S[i][j][1][start:stop])
               
                Dn[i][j][0] = S[0][0][0]          # Wavelength remains the same
                Dn[i][j][1] = S[i][j][1]*R[i][j]  # Multiply flux by ratio
                Dn[i][j][2] = S[i][j][2]*R[i][j]  # Multiply err by ratio
                Dn[i][j][3] = S[i][j][3]          # Dates remain the same

                #Dn[i][j][1] = (Dn[i][j][1] / np.median(Dn[i][j][1]))
                #Dn[i][j][2] = (Dn[i][j][2] / np.median(Dn[i][j][1]))


        print("Done")
        return np.array(Dn)

    def SaveData(self, data, name, location, compress):
        if compress == True:
            np.savez_compressed(name,data)
        else:
            np.savez(name,data)

        return None

    def LoadData(self,npz_file):
        data_file = np.load(npz_file, allow_pickle=True, encoding = 'bytes')
        data_file = data_file.f.arr_0
        return np.array(data_file)

    def BinXY(self,x,y,bin_pnts):
        if bin_pnts == 1:
            return x, y
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                bin_size    = int(len(x)/bin_pnts)
                bins        = np.linspace(x[0], x[-1], bin_size)
                digitized   = np.digitize(x, bins)
                bin_y       = np.array([y[digitized == i].mean() for i in range(0, len(bins))])
                return bins, bin_y


    def BinXYE(self,x,y,e,bin_pnts):
        
        if bin_pnts == 1:
            return x, y, e
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                bin_size    = int(len(x)/bin_pnts)
                bins        = np.linspace(x[0], x[-1], bin_size)
                digitized   = np.digitize(x, bins)
                bin_y       = np.array([y[digitized == i].mean() for i in range(0, len(bins))])
                bin_e       = np.array([np.linalg.norm(e[digitized == i])/len(e[digitized == i]) for i in range(0, len(bins))])
                return bins, bin_y, bin_e


    def Wave2RV(self,Wave,rest_wavelength,RV_BP):
        c = 299792458
        rest_wavelength = rest_wavelength*(RV_BP*1.e3)/c + rest_wavelength # Convert to beta pic reference frame
        delta_wavelength = Wave-rest_wavelength
        RV = ((delta_wavelength/rest_wavelength)*c)/1.e3    # km/s
        return RV

    def ShiftDATA(self, DATA, units, part, epoch):
        # Create an arrary of zeros with a length equal to the
        # number of pixels (units) by which the spectrum will be shifted.
        # Adding 1e-20 to avoid division by 0.
        zeros   = np.zeros(units)+1e-20
        
        # For observations done before 2017
        if epoch == 'pre':
            # Add an array of zeros to the end of the data
            D_F = np.concatenate((DATA[1],zeros))
            D_E = np.concatenate((DATA[2],zeros))
            if part == 'B':
                # I concatenate a fixed length array after the data array to ensure
                # the final array is equally long for both detector B and A.
                # The spacing of this array is not important, just the length
                # so I have arbitrarily put a range from the end of
                # the data to DATA[0][-1]+5.0.
                D_W = np.concatenate((DATA[0],np.linspace(DATA[0][-1],DATA[0][-1]+40.0,units)))
            if part == 'A':
                D_W = np.concatenate((DATA[0],np.linspace(DATA[0][-1],DATA[0][-1]+40.0,units)))
        else:
            if part == 'B':
                D_W = np.concatenate((np.linspace(DATA[0][0]-40.0,DATA[0][0],units),DATA[0]))  
            if part == 'A':
                D_W = np.concatenate((np.linspace(DATA[0][0]-40.0,DATA[0][0],units),DATA[0]))     
            # Add an array of zeros to the start of the data
            D_F = np.concatenate((zeros,DATA[1]))
            D_E = np.concatenate((zeros,DATA[2]))
        
        return D_W, D_F, D_E

    def RadCalc(self,accel):
        acc = accel*1000/60
        return np.sqrt((6.674e-11*1.75*1.9884e30)/acc)/(1.8*6.957e8)

class Model:

    def voigt_wofz(self, a, u):

        ''' Compute the Voigt function using Scipy's wofz().

        # Code from https://github.com/nhmc/Barak/blob/\
        087602fb372812c0603441ca1ce6820e96963b88/barak/absorb/voigt.py
        
        Explanation: https://nhmc.github.io/Barak/generated/barak.voigt.voigt.html#barak.voigt.voigt

        Parameters
        ----------
        a: float
          Ratio of Lorentzian to Gaussian linewidths.
        u: array of floats
          The frequency or velocity offsets from the line centre, in units
          of the Gaussian broadening linewidth.

        See the notes for `voigt` for more details.
        '''
        '''
        This code has been removed for speed purposes
        try:
             from scipy.special import wofz
        except ImportError:
             s = ("Can't find scipy.special.wofz(), can only calculate Voigt "
                  " function for 0 < a < 0.1 (a=%g)" % a)  
             print(s)
        else:
             return wofz(u + 1j * a).real
        '''
        return wofz(u + 1j * a).real

    def absorption(self, l, v_comp, N, vturb, T, species, param):
        '''
        v_comp --> The component speed in km/s
        N --> The column density exponent (10**N)
        vturb --> Micro turbulence
        T --> Gas temperature
        species --> Given in params.json
        '''

        w       = param["lines"]["line"][species]["Wavelength"]
        mass    = param["lines"]["line"][species]["Mass"]
        fosc    = param["lines"]["line"][species]["Strength"]
        delta   = param["lines"]["line"][species]["Gamma"] /(4.*np.pi)
        N_col   = np.array([1.])*10**N

        c_light     = 2.99793e14        # Speed of light
        k           = 1.38064852e-23    # Boltzmann constant in J/K = m^2*kg/(s^2*K) in SI base units
        u           = 1.660539040e-27   # Atomic mass unit (Dalton) in kg
        feature  = np.ones(len(l))

        b_wid   = np.sqrt((T/mass) + ((vturb/0.12895223)**2))
        b       = 4.30136955e-3*b_wid
        dnud    = b*c_light/w
        xc      = l/(1.+v_comp*1.e9/c_light) 
        v       = 1.e4*abs(((c_light/xc)-(c_light/w))/dnud)
        tv      = 1.16117705e-14*N_col*w*fosc/b_wid
        a       = delta/dnud
        hav     = tv*self.voigt_wofz(a,v)

        # To avoid calculating super tiny numbers
        for j in range(len(hav)):

            if hav[j] < 50:      
                feature[j]  =   feature[j]*np.exp(-hav[j])       
            else:
                feature[j]  =   0.

        return feature
    
    def absorptionSiIV(self, l, v_comp, N, vturb, T):
        '''
        v_comp --> The component speed in km/s
        N --> The column density exponent (10**N)
        vturb --> Micro turbulence
        T --> Gas temperature
        species --> Given in params.json
        '''

        home = "/home/pas/science/exocomets"

        with open(home+'/params.json') as param_file:    
            param = json.load(param_file)

        species = "SiIV"

        w       = param["lines"]["line"][species]["Wavelength"]
        mass    = param["lines"]["line"][species]["Mass"]
        fosc    = param["lines"]["line"][species]["Strength"]
        delta   = param["lines"]["line"][species]["Gamma"] /(4.*np.pi)
        N_col   = np.array([1.])*10**N

        c_light     = 2.99793e14        # Speed of light
        k           = 1.38064852e-23    # Boltzmann constant in J/K = m^2*kg/(s^2*K) in SI base units
        u           = 1.660539040e-27   # Atomic mass unit (Dalton) in kg
        feature  = np.ones(len(l))

        b_wid   = np.sqrt((T/mass) + ((vturb/0.12895223)**2))
        b       = 4.30136955e-3*b_wid
        dnud    = b*c_light/w
        xc      = l/(1.+v_comp*1.e9/c_light) 
        v       = 1.e4*abs(((c_light/xc)-(c_light/w))/dnud)
        tv      = 1.16117705e-14*N_col*w*fosc/b_wid
        a       = delta/dnud
        hav     = tv*self.voigt_wofz(a,v)

        # To avoid calculating super tiny numbers
        for j in range(len(hav)):

            if hav[j] < 50:      
                feature[j]  =   feature[j]*np.exp(-hav[j])       
            else:
                feature[j]  =   0.

        LSF_kernel = self.LSF(l, param["lines"]["line"]["SiIV"]["Wavelength"], home)

        convolved_abs = np.convolve(feature, LSF_kernel, mode='same')

        return convolved_abs

    def LSF(self, W, lsf_cen, home_dir):
        ''' Tabulated Theoretical Line Spread Functions at Lifetime position 3
        See https://www.stsci.edu/hst/instrumentation/cos/performance/spectral-resolution
        '''
        # Load the wavelengths which have computed LSF values
        X               =  np.genfromtxt(home_dir+'/src/aa_LSFTable_G130M_1291_LP3_cn.dat', unpack=True).T[0]
        
        # Find the LSF closest to lsf_cen in Angstrom.
        closest_LSF     = min(X, key=lambda x:abs(x-lsf_cen))

        df              = pd.read_csv(home_dir+'/src/aa_LSFTable_G130M_1291_LP3_cn.dat', delim_whitespace=True)
        y_LSF_tabu      = df[0:][str(int(closest_LSF))]
        
        # We re-center the LSF on 0
        x_LSF_tabu    = np.arange(-len(y_LSF_tabu)/2,len(y_LSF_tabu)/2)

        # We calculate how many Ångstrom a COS wavelength step is
        dw_cos          = (W[-1]-W[0])/len(W)

        # We wish to calculate the linespread function over our wavelength range,
        # which we shift so that the line is centered
        x_LSF_kernel_cos  = W-lsf_cen

        # We convert the tabulated values to the COS wavelenght scale
        x_LSF_tabu_cos  = x_LSF_tabu*dw_cos
        y_LSF_tabu_cos  = y_LSF_tabu*dw_cos

        # Note that the input spectrum must be broader than the LSF kernel (321 pixels)
        # So in other words len(x_LSF_kernel_cos) > len(x_LSF_tabu_cos) must be true
        # If it is not the case, you must select a broader wavelength range.
        LSF_kernel      = np.interp(x_LSF_kernel_cos, x_LSF_tabu_cos, y_LSF_tabu)

        #We normalise the LSF kernel
        LSF_kernel      = LSF_kernel/np.sum(LSF_kernel)

        return LSF_kernel

class Stats:
    '''
    A collection of statistical functions.
    '''
        
    def chi2(self, X):
        '''
        Calculated the Chi2 value
        X[0] => Obs
        X[1] => Err
        X[2] => Model
        '''
        return np.sum(((X[0] - X[2]) / X[1])**2.)

    def chi2_lm(self, params, F, E, Const, ModelType, param):
        model = m.Model(params, Const, ModelType, param)[0]
        return (model - F)**2 / E**2
    
    def Merit(self, X):
        ''' Given a Chi2 value
        we calculate a likelihood as our merit function
        '''
        return np.exp(-self.chi2(X)/2.)

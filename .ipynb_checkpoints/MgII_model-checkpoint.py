from astropy.modeling.models import Voigt1D
from numpy.polynomial import Polynomial
from lyapy import voigt ## this is my voigt.py file
import numpy as np
from lyapy import lyapy
from lmfit import Model
import pandas as pd
import matplotlib.pyplot as plt
from lmfit.model import save_modelresult, load_modelresult, save_model
from matplotlib import rc
import configparser

plt.ion()

mgii_k_line_center = 2796.3543 # Angstroms, vacuum wavelength, reference: Morton 2003
c_kms = 2.99792458e5 # km/s, speed of light
c_km = 2.99792458e5 # km/s, speed of light
ccgs = 2.99792458e10 # cm/s, speed of light
me = 9.1093897e-28 # g, electron mass
e = 4.8032068e-10 # esu, electron charge

rc('font',**{'family':'sans-serif'})
rc('text', usetex=True)

import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size


rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

rc('text.latex', preamble=r'\usepackage[helvet]{sfmath}')

def main(config_filename):

    global config
    config = load_config_file(config_filename)

    # read model into lmfit
    model = Model(my_model)

    # read in parameter hints from config file
    model, params = get_parameter_hints(model)

    # read in data
    wavelength_array, flux_array, error_array = read_data()

    ## evaluate initial model
    initial_model_profile = model.eval(params, wavelength_array=wavelength_array)

    ## perform the fit
    result = model.fit(flux_array, wavelength_array=wavelength_array, weights=1./error_array)
    print(result.fit_report())

    make_diagnostic_fit_figure(wavelength_array,flux_array,error_array,initial_model_profile,result)

    make_and_save_fit_figure(wavelength_array,flux_array,error_array,result)

    save_fit_results(wavelength_array, flux_array, error_array, result)


    return

def load_config_file(config_filename):

    config = configparser.ConfigParser()
    config.read(config_filename)
    
    return config

def get_parameter_hints(model):

    model.set_param_hint('vs',value=config['Parameter hints'].getfloat('vs value'), 
               min=config['Parameter hints'].getfloat('vs min'), 
               max=config['Parameter hints'].getfloat('vs max'), 
               vary=config['Parameter hints'].getboolean('vs vary')) 
    model.set_param_hint('am',value=config['Parameter hints'].getfloat('am value'), 
               min=config['Parameter hints'].getfloat('am min'), 
               max=config['Parameter hints'].getfloat('am max'), 
               vary=config['Parameter hints'].getboolean('am vary')) 
    model.set_param_hint('fw_L',value=config['Parameter hints'].getfloat('fw_L value'), 
               min=config['Parameter hints'].getfloat('fw_L min'), 
               max=config['Parameter hints'].getfloat('fw_L max'), 
               vary=config['Parameter hints'].getboolean('fw_L vary')) 
    model.set_param_hint('fw_G',value=config['Parameter hints'].getfloat('vs value'), 
               min=config['Parameter hints'].getfloat('fw_G min'), 
               max=config['Parameter hints'].getfloat('fw_G max'), 
               vary=config['Parameter hints'].getboolean('fw_G vary')) 
    model.set_param_hint('p',value=config['Parameter hints'].getfloat('p value'), 
               min=config['Parameter hints'].getfloat('p min'), 
               max=config['Parameter hints'].getfloat('p max'), 
               vary=config['Parameter hints'].getboolean('p vary')) 
    model.set_param_hint('vs_rev',value=config['Parameter hints'].getfloat('vs_rev value'), 
               min=config['Parameter hints'].getfloat('vs_rev min'), 
               max=config['Parameter hints'].getfloat('vs_rev max'), 
               vary=config['Parameter hints'].getboolean('vs_rev vary')) 

    model.set_param_hint('mg2_col',value=config['Parameter hints'].getfloat('mg2_col value'), 
               min=config['Parameter hints'].getfloat('mg2_col min'), 
               max=config['Parameter hints'].getfloat('mg2_col max'), 
               vary=config['Parameter hints'].getboolean('mg2_col vary')) 
    model.set_param_hint('mg2_b',value=config['Parameter hints'].getfloat('mg2_b value'), 
               min=config['Parameter hints'].getfloat('mg2_b min'), 
               max=config['Parameter hints'].getfloat('mg2_b max'), 
               vary=config['Parameter hints'].getboolean('mg2_b vary')) 
    model.set_param_hint('mg2_vel',value=config['Parameter hints'].getfloat('mg2_vel value'), 
               min=config['Parameter hints'].getfloat('mg2_vel min'), 
               max=config['Parameter hints'].getfloat('mg2_vel max'), 
               vary=config['Parameter hints'].getboolean('mg2_vel vary')) 

    model.set_param_hint('mg2_col2',value=config['Parameter hints'].getfloat('mg2_col2 value'), 
               min=config['Parameter hints'].getfloat('mg2_col2 min'), 
               max=config['Parameter hints'].getfloat('mg2_col2 max'), 
               vary=config['Parameter hints'].getboolean('mg2_col2 vary')) 
    model.set_param_hint('mg2_b2',value=config['Parameter hints'].getfloat('mg2_b2 value'), 
               min=config['Parameter hints'].getfloat('mg2_b2 min'), 
               max=config['Parameter hints'].getfloat('mg2_b2 max'), 
               vary=config['Parameter hints'].getboolean('mg2_b2 vary')) 
    model.set_param_hint('mg2_vel2',value=config['Parameter hints'].getfloat('mg2_vel2 value'), 
               min=config['Parameter hints'].getfloat('mg2_vel2 min'), 
               max=config['Parameter hints'].getfloat('mg2_vel2 max'), 
               vary=config['Parameter hints'].getboolean('mg2_vel2 vary')) 

    model.set_param_hint('c0',value=config['Parameter hints'].getfloat('c0 value'), 
               min=config['Parameter hints'].getfloat('c0 min'), 
               max=config['Parameter hints'].getfloat('c0 max'), 
               vary=config['Parameter hints'].getboolean('c0 vary')) 
    model.set_param_hint('c1',value=config['Parameter hints'].getfloat('c1 value'), 
               min=config['Parameter hints'].getfloat('c1 min'), 
               max=config['Parameter hints'].getfloat('c1 max'), 
               vary=config['Parameter hints'].getboolean('c1 vary')) 
    model.set_param_hint('c2',value=config['Parameter hints'].getfloat('c2 value'), 
               min=config['Parameter hints'].getfloat('c2 min'), 
               max=config['Parameter hints'].getfloat('c2 max'), 
               vary=config['Parameter hints'].getboolean('c2 vary')) 
    model.set_param_hint('c3',value=config['Parameter hints'].getfloat('c3 value'), 
               min=config['Parameter hints'].getfloat('c3 min'), 
               max=config['Parameter hints'].getfloat('c3 max'), 
               vary=config['Parameter hints'].getboolean('c3 vary')) 
    model.set_param_hint('c4',value=config['Parameter hints'].getfloat('c4 value'), 
               min=config['Parameter hints'].getfloat('c4 min'), 
               max=config['Parameter hints'].getfloat('c4 max'), 
               vary=config['Parameter hints'].getboolean('c4 vary')) 

    params = model.make_params()
    model.print_param_hints()

    return model, params

def read_data():

    data_path = config['Star and data properties'].get('data path')
    star_name = config['Star and data properties'].get('star name')

    data = pd.read_table(data_path + star_name + '.txt', sep=' ', names=['wavelength','flux','error'])
    wavelength_array = data['wavelength']
    flux_array = data['flux']
    error_array = data['error']

    ## mask the data - Isolate the Mg II line
    wavelength_min = config['Star and data properties'].getfloat('wavelength min')
    wavelength_max = config['Star and data properties'].getfloat('wavelength max')
    mask = (wavelength_array > wavelength_min) & (wavelength_array < wavelength_max) 
    wavelength_array = np.array(wavelength_array[mask])
    flux_array = np.array(flux_array[mask])
    error_array = np.array(error_array[mask])

    return wavelength_array, flux_array, error_array

def make_diagnostic_fit_figure(wavelength_array,flux_array,error_array,initial_model_profile,result):
    star_name = config['Star and data properties'].get('star name')

    plt.figure()
    plt.errorbar(wavelength_array,flux_array,yerr=error_array,color='k',label='data')
    plt.plot(wavelength_array,initial_model_profile,color='gray',linestyle='--')
    plt.plot(wavelength_array, result.best_fit, color='deeppink')
    plt.title('Diagnostic fit plot for ' + star_name, fontsize=18)
    plt.xlabel('Wavelength (A)',fontsize=18)
    plt.ylabel('Flux Density (erg/cm2/s/A)',fontsize=18)
    plt.tight_layout()

    return

def make_and_save_fit_figure(wavelength_array,flux_array,error_array,result):

    save_path = config['Star and data properties'].get('save path')
    star_name = config['Star and data properties'].get('star name')

    plt.figure()
    plt.errorbar(wavelength_array,flux_array,yerr=error_array,color='k',label='data')
    plt.plot(wavelength_array, result.best_fit, color='deeppink')
    plt.xlabel('Wavelength (A)',fontsize=18)
    plt.ylabel('Flux Density (erg/cm2/s/A)',fontsize=18)
    plt.title(star_name,fontsize=18)
    plt.tight_layout()
    plt.savefig(save_path+star_name+'_bestfit.png')

    return

def save_fit_results(wavelength_array, flux_array, error_array, result):

    save_path = config['Star and data properties'].get('save path')
    star_name = config['Star and data properties'].get('star name')

    with open(save_path+star_name+'_result.txt', 'w') as fh:
        fh.write(result.fit_report())

    stellar_intrinsic_profile, continuum_profile, ism_attenuation, ism_attenuation2, \
            stellar_observed_profile, stellar_observed_profile_convolved = \
                                        my_model(wavelength_array, vs=result.best_values['vs'], 
                                        am=result.best_values['am'], fw_L=result.best_values['fw_L'], 
                                        fw_G=result.best_values['fw_G'], p=result.best_values['p'], 
                                        vs_rev=result.best_values['vs_rev'], mg2_col=result.best_values['mg2_col'], 
                                        mg2_b=result.best_values['mg2_b'], mg2_vel=result.best_values['mg2_vel'], 
                                        mg2_col2=result.best_values['mg2_col2'], mg2_b2=result.best_values['mg2_b2'], 
                                        mg2_vel2=result.best_values['mg2_vel2'], c0=result.best_values['c0'], 
                                        c1=result.best_values['c1'], c2=result.best_values['c2'], 
                                        c3=result.best_values['c3'], c4=result.best_values['c4'], fitting=False)


    ## save data and fit lines to file
    df_to_save = pd.DataFrame(data={'wavelength_array': wavelength_array, 
                                'flux_array': flux_array,
                                'error_array': error_array,
                                'best_fit_model': result.best_fit,
                                'stellar_observed_profile_convolved': stellar_observed_profile_convolved, # this should be identical to result.best_fit
                                'stellar_observed_profile': stellar_observed_profile,
                                'continuum_profile': continuum_profile,
                                'ism_attenuation': ism_attenuation,
                                'ism_attenuation2': ism_attenuation2,
                                'stellar_intrinsic_profile': stellar_intrinsic_profile
                                })
    df_to_save.to_csv(save_path+star_name+'_bestfit_lines.csv')

    return


def my_model(wavelength_array, vs, am, fw_L, fw_G, p, vs_rev, mg2_col, mg2_b, mg2_vel, mg2_col2, mg2_b2, mg2_vel2, c0, c1, c2, c3, c4, fitting=True, convolve=True): 
    
    print(vs, am, fw_L, fw_G, p, vs_rev, mg2_col, mg2_b, mg2_vel, mg2_col2, mg2_b2, mg2_vel2, c0, c1, c2, c3, c4)


    ##### constructing the intrinsic stellar emission line ##################################################################
    stellar_intrinsic_profile = intrinsic_stellar_emission_line(wavelength_array, vs, am, fw_L, fw_G, p, vs_rev)
    #########################################################################################################################

    #### constructing the stellar continuum ################################################################################
    continuum_profile = make_continuum_profile(wavelength_array, c0, c1, c2, c3, c4)
    ########################################################################################################################

    #### construct the observed stellar emission line (i.e., attenuated by the ISM)##########################################
    #print('my_model mg2_b, mg2_b2 = ' + str(mg2_b) + ', '+ str(mg2_b2))
    ism_attenuation = total_tau_profile_func_mgii(wavelength_array,mg2_col, mg2_b, mg2_vel,which_line='k')
    ism_attenuation2 = total_tau_profile_func_mgii(wavelength_array,mg2_col2, mg2_b2, mg2_vel2,which_line='k')

    stellar_observed_profile = attenuate_stellar_emission_line(stellar_intrinsic_profile + continuum_profile, ism_attenuation * ism_attenuation2)

    if convolve:
        resolution = make_resolution_variable(wavelength_array)
        stellar_observed_profile_convolved = np.convolve( stellar_observed_profile, resolution, mode='same')
    else:
        stellar_observed_profile_convolved = stellar_observed_profile.copy()
    #########################################################################################################################

    if fitting:
    
            return stellar_observed_profile_convolved
          
    else:
        
        return stellar_intrinsic_profile, continuum_profile, ism_attenuation, ism_attenuation2, stellar_observed_profile, stellar_observed_profile_convolved


def make_resolution_variable(wavelength_array, index=1, lsf_filename='STIS_E230H_240nm_LSF.dat'):
    lsf_path = config['Star and data properties'].get('lsf path')
    stis_lsf = np.loadtxt(lsf_path+lsf_filename,skiprows=2)  
    stis_dispersion = wavelength_array[1]-wavelength_array[0]
    resolution = lyapy.ready_stis_lsf(stis_lsf[:,0],stis_lsf[:,index],stis_dispersion,wavelength_array)
    return resolution

def intrinsic_stellar_emission_line(wavelength_array, vs, am, fw_L, fw_G, p, vs_rev):

    line_center = (vs/c_kms + 1.) * mgii_k_line_center # convert stellar radial velocity (km/s) to wavelength units (Angstroms)
    line_center_rev = ((vs + vs_rev)/c_kms + 1.) * mgii_k_line_center # convert self-reversal radial velocity (km/s) to wavelength units (Angstroms)

    sigma_G = fw_G/c_kms * mgii_k_line_center # convert Gaussian FWHM of the Voigt profile from km/s to Angstroms
    sigma_L = fw_L/c_kms * mgii_k_line_center # convert Lorentzian FWHM of the Voigt profile from km/s to Angstroms
                                           

    voigt_profile_func = Voigt1D(x_0 = line_center, amplitude_L = 10**am, fwhm_L = sigma_L, fwhm_G = sigma_G)

    voigt_profile_func_rev = Voigt1D(x_0 = line_center_rev, amplitude_L = 10**am, fwhm_L = sigma_L, fwhm_G = sigma_G)

    self_reversal =  np.exp(-p * voigt_profile_func_rev(wavelength_array) / np.max(voigt_profile_func_rev(wavelength_array)))

    stellar_intrinsic_profile = voigt_profile_func(wavelength_array) * self_reversal

    return stellar_intrinsic_profile


def attenuate_stellar_emission_line(stellar_intrinsic_profile, ism_attenuation_profile):

    return stellar_intrinsic_profile * ism_attenuation_profile

def make_continuum_profile(wavelength_array, c0, c1, c2, c3, c4):

    polynomial_func = Polynomial(coef=(c0, c1, c2, c3, c4), domain = [np.min(wavelength_array), np.max(wavelength_array)])
    continuum_profile = polynomial_func(wavelength_array)

    return continuum_profile



def total_tau_profile_func_mgii(wavelength_array,col,b,vel,which_line):

    """
    Given a wavelength array and parameters (column density, b value, and 
    velocity centroid), computes the Voigt profile of Mg II
    and returns the combined or single absorption profile.

    """
    
    
    #print('total_tau_profile_func_mgii col, b, vel = ' + str(col) + ', '+ str(b) + ', '+ str(vel))

    ##### ISM absorbers #####
   
    wave_all,tau_all=tau_profile_mgii(col,vel,b,'k')
    tau_k=np.interp(wavelength_array,wave_all,tau_all)

    wave_all,tau_all=tau_profile_mgii(col,vel,b,'h')
    tau_h=np.interp(wavelength_array,wave_all,tau_all)

    #hwave_all3,htau_all3=tau_profile_oi(h1_col,h1_vel,h1_b,'1259.518')
    #tauh1_3=np.interp(wave_to_fit,hwave_all3,htau_all3)


    ## Adding the optical depths and creating the observed profile ##

    if which_line == 'k':
      tot_tau = tau_k
    elif which_line == 'h':
      tot_tau = tau_h
    else:
      print("uh oh something's wrong")
    tot_ism = np.exp(-tot_tau)

    return tot_ism


def tau_profile_mgii(ncols,vshifts,vdop,which_line):

    """ 
    Computes a MgII Voigt profile for given column density (log units, ncols parameter),
    velocity centroid (km/s, vshifts parameter), and b parameter (vdop, km/s).

    """

    if which_line == 'k':
        lam0s,fs,gammas=2796.3543,6.155E-01,2.625E+08 
    elif which_line == 'h':
        lam0s,fs,gammas=2803.5315,3.058E-01,2.595E+08
    else:
        raise ValueError("which_line can only equal 'k' or 'h'!")


    Ntot=10.**ncols  # column density of Mg II gas
    nlam=1000       # number of elements in the wavelength grid
    xsections_onesided=np.zeros(nlam)  # absorption cross sections as a 
                                       # function of wavelength (one side of transition)
    u_parameter=np.zeros(nlam)  # Voigt "u" parameter
    nu0s=ccgs/(lam0s*1e-8)  # wavelengths of Lyman alpha in frequency
    nuds=nu0s*vdop/c_km    # delta nus based off vdop parameter
    a_parameter = np.abs(gammas/(4.*np.pi*nuds) ) # Voigt "a" parameter -- damping parameter
    #if not 0 <= a_parameter <= 0.1:
    #    print(gammas, vdop, nuds, a_parameter)
    
    xsections_nearlinecenter = np.sqrt(np.pi)*(e**2)*fs*lam0s/(me*ccgs*vdop*1e13)  # cross-sections 
                                                                           # near Lyman line center

    wave_edge=lam0s-1.5 # define wavelength cut off
    wave_symmetrical=np.zeros(2*nlam-1) # huge wavelength array centered around a Lyman transition
    wave_onesided = np.zeros(nlam)  # similar to wave_symmetrical, but not centered 
                                    # around a Lyman transition 
    lamshifts=lam0s*vshifts/c_km  # wavelength shifts from vshifts parameter

    ## find end point for wave_symmetrical array and create wave_symmetrical array
    num_elements = 2*nlam - 1
    first_point = wave_edge
 
    mid_point = lam0s
    end_point = 2*(mid_point - first_point) + first_point
    wave_symmetrical = np.linspace(first_point,end_point,num=num_elements)
    wave_onesided = np.linspace(lam0s,wave_edge,num=nlam)

    freq_onesided = ccgs / (wave_onesided*1e-8)  ## convert "wave_onesided" array to a frequency array

    u_parameter = (freq_onesided-nu0s)/nuds  ## Voigt "u" parameter -- dimensionless frequency offset

    xsections_onesided=xsections_nearlinecenter*voigt.voigt(a_parameter,u_parameter)  ## cross-sections
                                                                                # single sided
                                                                                ## can't do symmetrical 

    xsections_onesided_flipped = xsections_onesided[::-1]
    
    ## making the cross-sections symmetrical
    xsections_symmetrical=np.append(xsections_onesided_flipped[0:nlam-1],xsections_onesided) 
    deltalam=np.max(wave_symmetrical)-np.min(wave_symmetrical)
    dellam=wave_symmetrical[1]-wave_symmetrical[0] 
    nall=np.round(deltalam/dellam)
    wave_all=deltalam*(np.arange(nall)/(nall-1))+wave_symmetrical[0]

    tau_all = np.interp(wave_all,wave_symmetrical+lamshifts,xsections_symmetrical*Ntot)

    return wave_all,tau_all

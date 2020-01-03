# Baptiste GROSS
# December 26th 2019
# Dreem Sleep Stages Classification Challenge 

from imports import *

# features computation

def basic_signal_features(signal, signal_name=None):
    """
    Computes basic features for time series analysis (absolute mean/min/max/median/sd)
    signal is a 2D array (signals X time points)
    """
    
    res = []
    
    res.append(abs(np.mean(signal, axis=1)))
    res.append(abs(np.median(signal, axis=1)))
    res.append(np.std(signal, axis=1))
    res.append(abs(np.min(signal, axis=1)))
    res.append(abs(np.max(signal, axis=1)))
        
    if signal_name:
        names = []
        names.append(signal_name + '_abs_mean')
        names.append(signal_name + '_abs_median')
        names.append(signal_name + '_std')
        names.append(signal_name + '_abs_min')
        names.append(signal_name + '_abs_max')
    
    if signal_name:return res, names
    else:return res

    
def chaos_features(signal, signal_name=None):
    """
    Chaos theory features 
    signal is a 2D array (signals X time points)
    """ 
    
    res = []
    
    signal_df = pd.DataFrame(signal)
    
    res.append(signal_df.apply(entropy.petrosian_fd, axis=1))
    res.append(signal_df.apply(entropy.app_entropy, axis=1))
    res.append(signal_df.apply(entropy.higuchi_fd, axis=1))
    
    if signal_name:
        names = []
        names.append(signal_name + '_fd_petrosian')
        names.append(signal_name + '_app_entropy')
        names.append(signal_name + '_higuchi_fd')
        
    if signal_name:return res, names
    else:return res

    
def euclidean_distance_eeg(databases, eegs_name):
    """
    Computes Euclidean distance between EEGs signals 2 by 2
    databases is a dictionnary with eegs_name as the keys and EEGs signals as array (signals X time points) 
    """
    
    res = []

    combis = np.triu_indices(7, k=1)
    combis_names = ['diff_eeg_{}_{}'.format(combis[0][i]+1, combis[1][i]+1) for i in range(0, len(combis[0]))]
    
    for idx in range(0,databases[eegs_name[0]].shape[0]):
        vec = np.asarray([databases[eeg][idx] for eeg in eegs_name])
        res.append(metrics.pairwise_distances(vec)[combis])
        
    diff = pd.DataFrame(np.asarray(res), columns=combis_names)
        
    return(diff)


def frequency_features(signal, eeg_sig=False, signal_name=None):
    """
    Computes features coming from the signal periodograms 
    signal is a 2D array
    """ 
    
    freq = int(len(signal[0])/30) # 30 second window
    print('Sampling frequency is {} Hz'.format(freq))
    
    res = []
    
    f, s = sg.periodogram(signal, fs=freq)    
    
    res.append(f[np.argmax(s, axis=1)])
    res.append(np.amax(s, axis=1))
    res.append(np.sum(s, axis=1))
    
    
    if signal_name:
        names = []
        names.append(signal_name + '_max_freq')
        names.append(signal_name + '_max_spectrum')
        names.append(signal_name + '_sum_spectrum')
    
    
    if eeg_sig: # if the signal is a EEG, compute the bandpower of the alpha, beta, theta and gamma waves 
        
        res.append(np.sum(s[:, np.asarray(np.where((f > 0.5)&(f<4))[0])], axis=1))
        res.append(np.sum(s[:, np.asarray(np.where((f >= 4)&(f<8))[0])], axis=1))
        res.append(np.sum(s[:, np.asarray(np.where((f >= 8)&(f<12))[0])], axis=1))
        res.append(np.sum(s[:, np.asarray(np.where((f >= 12)&(f<30))[0])], axis=1))
        
        if signal_name:
            
            names.append(signal_name + '_sum_delta')
            names.append(signal_name + '_sum_theta')
            names.append(signal_name + '_sum_alpha')
            names.append(signal_name + '_sum_beta')

            
    if signal_name:return res, names
    else: return res

    
def decomposition_features(signal, signal_name=None):
    """
    Computes basic features as well as autoregressive coefficients calculated on the trend and residual components 
    from the decomposition of the signal.
    signal is a 2D array
    """
    
    res = []

    for i in range(0,len(signal)) :
         
        # decomposition
        
        sdc = seasonal_decompose(signal[i], model='additive', freq=int(len(signal[i])/30))
        
        # dropping NaN values left by the seasonal_decomposition function
        
        nan_trend = np.where(np.invert(np.isnan(sdc.trend)))[0]
        trend_sig = sdc.trend[nan_trend]
        
        nan_resid = np.where(np.invert(np.isnan(sdc.resid)))[0]
        resid_sig = sdc.resid[nan_resid]
    
        
        # basic features calculated on trend
    
        res += list(np.reshape(basic_signal_features(np.reshape(trend_sig, (1, len(trend_sig)))), (5,)))

       
        # basic features calculated on resid

        res += list(np.reshape(basic_signal_features(np.reshape(resid_sig, (1, len(resid_sig)))), (5,)))

        
        # autoregressive coefficients calculated on trend
        def ar_coefficients(sig):
            model = AR(sig)
            model = model.fit()
            return list(model.params)
    
        ar = ar_coefficients(trend_sig)
        res += ar


    res = list(np.reshape(res, (len(ar)+10, len(signal))))

    if signal_name:
        
        names = []
        names.append(signal_name + '_abs_mean_trend')
        names.append(signal_name + '_abs_median_trend')
        names.append(signal_name + '_std_trend')
        names.append(signal_name + '_abs_min_trend')
        names.append(signal_name + '_abs_max_trend')
        names.append(signal_name + '_abs_mean_resid')
        names.append(signal_name + '_abs_median_resid')
        names.append(signal_name + '_std_resid')
        names.append(signal_name + '_abs_min_resid')
        names.append(signal_name + '_abs_max_resid')
        names += ['{}_AR_{}'.format(signal_name, i) for i in range(len(ar))]
    
    
    if signal_name: return res, names
    else: return res


def compute_features(database, biomarkers):
    """
    """
    
    val = []
    nme = []
    eegs = ['eeg_{}'.format(i) for i in range(1,8)]
    
    for marker in biomarkers:
        # basic features
        res, names = basic_signal_features(database[marker], signal_name=marker)
        
        # chaos features
        chaos_res, chaos_names = chaos_features(database[marker], signal_name=marker)
        
        # frequency_features
        eeg_sig = marker in eegs
        freq_res, freq_names = frequency_features(database[marker], signal_name=marker, eeg_sig=eeg_sig)
        
        # decomposition features
        dec_res, dec_names = decomposition_features(database[marker], signal_name=marker)
        
        res += chaos_res
        res += freq_res
        res += dec_res
        names += chaos_names
        names += freq_names
        names += dec_names
        
        val += res
        nme += names
           
    dist_df = euclidean_distance_eeg(database, eegs)
    feature_df = pd.DataFrame(np.vstack(val).T, columns=nme)
    
    return pd.concat([feature_df, dist_df], axis=1)
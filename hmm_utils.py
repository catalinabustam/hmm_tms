def get_time_series(dirpath, subects, stimulus, scan, parcel):
    
    subject_list = []
    series_array = np.empty([0,15, 600])
    
    for subject in subjects:
        file_path = f'{dirpath}/{subject}/{stimulus}/roi_timeseries_0/{scan}/_compcor_ncomponents_5_selector_pc10.linear1.wm1.global0.motion1.quadratic1.gm0.compcor1.csf1/_bandpass_freqs_0.01.0.1/{parcel}/roi_stats.csv'
        series = pd.read_csv(file_path, skiprows=[0,1], header=None, delimiter='\t')
        series = series.T
        series['network'] = schafer['network']
        series_average = series.groupby('network').mean()

        series_average = series_average.drop("Limbic", axis=0)
        
        scaled_series = pd.DataFrame(scaler.fit_transform(series_average.T).T)
   
        scaled_rows = scaled_series.values
        
        series_array = np.concatenate((series_array, scaled_rows[np.newaxis,...]), axis=0)
    return series_array
    

def get_all_data(scans, stimuli,subjects, dirpath, parcel): 
    all_c = np.empty((0, 15, 600))

    for scan in scans:

        for stimulus in stimuli:
  
            stimulus_array = get_time_series(dirpath, subjects, stimulus, scan, parcel)
            all_c = np.append(all_c, stimulus_array, axis=0)
          
    return all_c
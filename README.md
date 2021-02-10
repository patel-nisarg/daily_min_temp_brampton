# Daily Minimum Temperature Brampton
### Modelling Brampton daily minimum temperature data with RNNs.

Raw data is taken from LAMPS lab at York University here:
http://lamps.math.yorku.ca/OntarioClimate/index_app_data.htm#/historicaldata
 
Data is provided as **8964** grid points throughout Ontario corresponding to lattitude and longitude positions described here:
http://lamps.math.yorku.ca/OntarioClimate/content/DataDownload/LAMPS_Ontario_8964Pts.csv

Fast fourier transform is used to determine periodicity of data. Since we are working with yearly data, we should see that when normalized per year, the most important frequency is 1.

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "")

Indeed, you can see from the step plot above that a frequency 1 year is the fundamental frequency followed by higher frequency harmonics.
To confirm, we can assert that np.argmax(fft) == 1 and check for any exceptions thrown.

The seasonality is then removed from the data and the model is trained on windowed data.

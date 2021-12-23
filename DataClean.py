import numpy as np

class  DataClean:
    
    def __init__(self):
        self.LimitHi = 0
        self.LimitLow = 0
        self.mdn = 0
       

    def remove_outlier(self, data):
        
        data=data.copy()
        self.LimitHi=np.mean(data) + 2*np.std(data)
        self.LimitLow=np.mean(data) - 2*np.std(data)
            
        data = data[(data < self.LimitHi) & (data > self.LimitLow)]
        
        return data

    def replace_outlier(self, data):
        
        data=data.copy()
        self.LimitHi=np.mean(data) + 2*np.std(data)
        self.LimitLow=np.mean(data) - 2*np.std(data)
        
        Flag_Good = (data <= self.LimitHi) & (data >= self.LimitLow)
        Flag_Bad = ~Flag_Good
        
        data[Flag_Bad] = np.mean(data[Flag_Good])
        
        return data
    
    
    # Finds string missing values i.e. '?' in otherwise numeric data, replaces
    # missing and converts to float
    def fill_median_str(self, data):
        
        data=data.copy()
        Flag_Good = np.array([element.isdigit() for element in data])
        Flag_Bad = ~Flag_Good
        
        self.mdn = data[Flag_Good]
        self.mdn = self.mdn.astype(float)
     
        data[Flag_Bad] = np.median(self.mdn)
        
        data = data.astype(float)
        
        return data
    
    # Replaces NAN with median values   
    def fill_median(self, data):
        
        data = data.copy()
        HasNan = np.isnan(data)
        
        if HasNan.sum()>0:
            data[HasNan] = np.nanmedian(data)
        
        return data
        
        
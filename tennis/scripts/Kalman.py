 # SimpleKalmanFilter - a Kalman Filter implementation for single variable models.
 # Created by Denys Sene, January, 1, 2017.
 # Released under MIT License - see LICENSE file for details.

import math
import numpy as np

class SimpleKalmanFilter:
    def start(self, mea_e, est_e, q):
        self._err_measure = mea_e
        self._err_estimate = est_e
        self._q = q
        self._last_estimate = 0
        self._current_estimate = 0
        self._kalman_gain = 0
        
    def updateEstimate(self, mea):
        self._kalman_gain = self._err_estimate/(self._err_estimate + self._err_measure)
        self._current_estimate = self._last_estimate + self._kalman_gain * (mea - self._last_estimate)
        self._err_estimate =  (1.0 - self._kalman_gain)*self._err_estimate + math.fabs(self._last_estimate-self._current_estimate)*self._q
        self._last_estimate = self._current_estimate
        return self._current_estimate

    def setMeasurementError(self, mea_e):
        self._err_measure = mea_e
    
    def setEstimateError(self, est_e):
        self._err_estimate=est_e
        
    def etProcessNoise(self, q):
        self._q=q
    
    def getKalmanGain(self):
        return self._kalman_gain
        
    def getEstimateError(self):
        return self._err_estimate

if __name__ == '__main__':
    kalman = SimpleKalmanFilter()
    kalman.start(1, 1, 1)
    for i in range(100):
        x = np.sin(i / 15.0)
        y = kalman.updateEstimate(x)
        print(y, ",")


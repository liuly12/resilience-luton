# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 10:09:12 2023

@author: leyan
"""

from wsimod.nodes.storage import Storage
from wsimod.core.constants import constants

class UnmeteredDemand(Storage):
    def __init__(self,
                 pollutant_load_conc = {},
                 **kwargs):
        """Node that accepts redundant water after supplying demand and filling the reservoir and generates raw wastewater loading.

        Args:
            pollutant_load_conc (float, optional):  concentration of raw wastewater [kg/m3]
        
        Functions intended to call in orchestration:
            distribute
        """
        #Update args
        self.pollutant_load_conc = pollutant_load_conc
        self.total_in_ = {}
        super().__init__(**kwargs)
        
        # revise mass balance - to account for the household load input
        # self.mass_balance_in.append(lambda: {pol:self.tank.storage['volume'] * self.pollutant_load_conc[pol] for pol in constants.ADDITIVE_POLLUTANTS} | {pol:self.pollutant_load_conc[pol] for pol in constants.NON_ADDITIVE_POLLUTANTS} | {'volume': 0})
        self.mass_balance_in = [lambda: self.total_in_]
    
    def create_demand(self):
        """Generate raw wastewater."""
        for key, item in self.pollutant_load_conc.items():
            if key in constants.ADDITIVE_POLLUTANTS:
                self.tank.storage[key] = item * self.tank.storage['volume']
            if key in constants.NON_ADDITIVE_POLLUTANTS:
                self.tank.storage[key] = item
        # for mass balance
        self.total_in_ = self.tank.storage
        
    	# Distribute any active storage
        storage = self.tank.pull_storage(self.tank.get_avail())
        retained = self.push_distributed(storage, of_type = ['Sewer', 'Waste'])
        _ = self.tank.push_storage(retained, force=True)
        if retained['volume'] > constants.FLOAT_ACCURACY:
            print("Unmetered demand ", self.name, " not able to push")
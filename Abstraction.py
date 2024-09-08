# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 10:09:12 2023

@author: leyan
"""

from wsimod.nodes.demand import Demand
from wsimod.core import constants

class Abstraction(Demand):
    def __init__(self,
                 pollutant_load_conc = {},
                 **kwargs):
        """Node that generates time-varying water demand specified by data input.

        Args:
            data_input_dict (dict, optional):  This must contains 'demand' along with the original 
            input vairables (e.g., 'temperature')
        
        Functions intended to call in orchestration:
            create_demand
        """
        #Update args
        self.pollutant_load_conc = pollutant_load_conc
        super().__init__(**kwargs)
    
    def get_demand(self):
        """Holder function to enable constant demand generation

        Returns:
            (dict): A VQIP that will contain constant demand
        """
        #TODO read/gen demand
        self.constant_demand = self.get_data_input('demand')
        pol = self.v_change_vqip(self.empty_vqip(), self.constant_demand)
        for key, item in self.pollutant_load_conc.items():
            if key in constants.ADDITIVE_POLLUTANTS:
                pol[key] = item * pol['volume']
            if key in constants.NON_ADDITIVE_POLLUTANTS:
                pol[key] = item
        return {'default' : pol}
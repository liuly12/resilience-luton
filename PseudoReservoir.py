# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 10:09:12 2023

@author: leyan
"""

from wsimod.nodes.storage import Storage

class PseudoReservoir(Storage):
    def __init__(self,
                 leakage_coeff = 0,
                 **kwargs):
        """Node that generates time-varying water demand specified by data input.

        Args:
            leakage_coeff (float, optional):  Proportion of water leaking into gw (before the demand 
	    abstract water from the pseudo-reservoir - manifested in calculation order in .ymal)
        
        Functions intended to call in orchestration:
            distribute
        """
        #Update args
        self.leakage_coeff = leakage_coeff
        super().__init__(**kwargs)
    
    def leak(self):
        """Leakage discharged to Groundwater_h node only."""
        # Distribute leakage to groundwater_h
        leakage = {'volume': self.tank.storage['volume'] * self.leakage_coeff}
        leakage = self.tank.pull_storage(leakage)
        retained = self.push_distributed(leakage, of_type = ['Groundwater_h'])
        _ = self.tank.push_storage(retained, force=True)

    def distribute(self):
        """Optional function that discharges all tank storage with push_distributed."""
	    # Distribute any active storage
        storage = self.tank.pull_storage(self.tank.get_avail())
        retained = self.push_distributed(storage, of_type = ['UnmeteredDemand'])
        _ = self.tank.push_storage(retained, force=True)
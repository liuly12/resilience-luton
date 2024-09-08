# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 10:50:55 2023

@author: bdobson
"""
from wsimod.core import constants
# import sys
# import os
# from tqdm import tqdm
# from math import log10
# from wsimod.nodes.land import ImperviousSurface
# from wsimod.nodes.nodes import QueueTank, Tank, ResidenceTank

def extensions(model):
    # decorate storm sewer to make it cso
    # for node_name, node in model.nodes.items():
    #     if any(ext in node_name for ext in ['storm', 'foul', 'combined']):
    #         node.make_discharge = wrapper_sewer(node)
    
    # decorate land->gw
    model.nodes['1823-land'].run = wrapper_land_gw(model.nodes['1823-land'])
    # model.nodes['1823-land'].surfaces[-1].push_to_sewers = wrapper_impervioussurface_sewer(model.nodes['1823-land'].surfaces[-1])
    
    model.nodes['1823-land'].surfaces[-1].atmospheric_deposition = adjust_atmospheric_deposition(model.nodes['1823-land'].surfaces[-1])
    model.nodes['1823-land'].surfaces[-1].precipitation_deposition = adjust_precipitation_deposition(model.nodes['1823-land'].surfaces[-1])
    model.nodes['1823-land'].surfaces[-1].inflows[0] = model.nodes['1823-land'].surfaces[-1].atmospheric_deposition
    model.nodes['1823-land'].surfaces[-1].inflows[1] = model.nodes['1823-land'].surfaces[-1].precipitation_deposition
    
    # Change model.run because new function has been introduced by the new node
    model.river_discharge_order = ['1823-river']
    # model.run = wrapper_run(model)
    # parameter adjustment
    # model.nodes['1823-land'].surfaces[-1].pore_depth = 0.1
    # model.nodes['1823-storm'].make_discharge = wrapper_stormsewer_cso(model.nodes['1823-storm'])
    
    # decorate fix head
    for node in ['6aead97c-0040-4e31-889d-01f628faf990',
                'fb3a7adf-ae40-4a9f-ad3f-55b3e4d5c6b7',
                '7e0cc125-fe7a-445b-af6b-bf55ac4065f9',
                'e07ddbc6-7158-4a47-b987-eb2b934dd257',
                'e4b324b5-60f9-48c2-9d64-d89d22a5305e',
                '88c7e69b-e4b3-4483-a438-0d6f9046cdee',
                'a057761f-e18e-4cad-84d4-9458edc182ef',
                '2b5397b7-a129-40a6-873d-cb2a0dd7d5b8'
                ]:
        model.nodes[node].end_timestep = end_timestep(model.nodes[node])
    
    # wq parameters for wwtw
    for wwtw, new_constants, variable, date in zip(['luton_stw-wwtw', 'luton_stw-wwtw', 'luton_stw-wwtw'],
                                                     [0.15, 0.08654, 0.0202/2.25],
                                                     ['phosphate', 'phosphate', 'phosphate'],
                                                     ['2000-01-01', '2001-01-01', '2005-01-01']):
        node = model.nodes[wwtw]
        node.end_timestep = wrapper_wwtw(node.end_timestep, node, variable, new_constants, date)

# set fixed head for 'ex-head' node
def end_timestep(self):
    self.mass_balance_in = [lambda: self.empty_vqip()]
    self.mass_balance_out = [lambda: self.empty_vqip()]
    self.mass_balance_ds = [lambda: self.empty_vqip()]
    def inner_function():
        """Update tank states & self.h
        """
        self.h = self.get_data_input('head')
        self.tank.storage['volume'] = (self.h - self.datum) * self.area * self.s
        self.tank.end_timestep()
        self.h = self.tank.get_head()
    return inner_function


def wrapper_land_gw(self):
    def run(#self
            ):
       """Call the run function in all surfaces, update surface/subsurface/
       percolation tanks, discharge to rivers/groundwater.
       """
       # Run all surfaces
       for surface in self.surfaces:
           surface.run()

       # Apply residence time to percolation
       percolation = self.percolation.pull_outflow()

       # Distribute percolation
       reply = self.push_distributed(percolation, of_type=["Groundwater_h"])

       if reply["volume"] > constants.FLOAT_ACCURACY:
           # Update percolation 'tank'
           _ = self.percolation.push_storage(reply, force=True)

       # Apply residence time to subsurface/surface runoff
       surface_runoff = self.surface_runoff.pull_outflow()
       subsurface_runoff = self.subsurface_runoff.pull_outflow()

       # Total runoff
       total_runoff = self.sum_vqip(surface_runoff, subsurface_runoff)
       if total_runoff["volume"] > 0:
           # Send to rivers (or nodes, which are assumed to be junctions)
           reply = self.push_distributed(total_runoff, of_type=["River_h", "Node"])

           # Redistribute total_runoff not sent
           if reply["volume"] > 0:
               reply_surface = self.v_change_vqip(
                   reply,
                   reply["volume"] * surface_runoff["volume"] / total_runoff["volume"],
               )
               reply_subsurface = self.v_change_vqip(
                   reply,
                   reply["volume"]
                   * subsurface_runoff["volume"]
                   / total_runoff["volume"],
               )

               # Update surface/subsurface runoff 'tanks'
               if reply_surface["volume"] > 0:
                   self.surface_runoff.push_storage(reply_surface, force=True)
               if reply_subsurface["volume"] > 0:
                   self.subsurface_runoff.push_storage(reply_subsurface, force=True)
    return run

def adjust_atmospheric_deposition(surface, ratio = 0.05):
    def atmospheric_deposition():
        """Inflow function to cause dry atmospheric deposition to occur, updating the 
        surface tank

        Returns:
            (tuple): A tuple containing a VQIP amount for model inputs and outputs 
                for mass balance checking. 
        """
        #TODO double check units in preprocessing - is weight of N or weight of NHX/noy?

        #Read data and scale
        nhx = surface.get_data_input_surface('nhx-dry') * surface.area * ratio
        noy = surface.get_data_input_surface('noy-dry') * surface.area * ratio
        srp = surface.get_data_input_surface('srp-dry') * surface.area * ratio

        #Assign pollutants
        vqip = surface.empty_vqip()
        vqip['ammonia'] = nhx
        vqip['nitrate'] = noy
        vqip['phosphate'] = srp

        #Update tank
        in_ = surface.dry_deposition_to_tank(vqip)

        #Return mass balance
        return (in_, surface.empty_vqip())
    return atmospheric_deposition

def adjust_precipitation_deposition(surface, ratio = 0.05):
    def precipitation_deposition():
        """Inflow function to cause wet precipitation deposition to occur, updating 
        the surface tank

        Returns:
            (tuple): A tuple containing a VQIP amount for model inputs and outputs 
                for mass balance checking. 
        """
        #TODO double check units - is weight of N or weight of NHX/noy?

        #Read data and scale
        nhx = surface.get_data_input_surface('nhx-wet') * surface.area * ratio
        noy = surface.get_data_input_surface('noy-wet') * surface.area * ratio
        srp = surface.get_data_input_surface('srp-wet') * surface.area * ratio

        #Assign pollutants
        vqip = surface.empty_vqip()
        vqip['ammonia'] = nhx
        vqip['nitrate'] = noy
        vqip['phosphate'] = srp

        #Update tank
        in_ = surface.wet_deposition_to_tank(vqip)

        #Return mass balance
        return (in_, surface.empty_vqip())
    return precipitation_deposition

def wrapper_wwtw(f, node,variable, value, date):
    def new_end_timestep():
        f()
        if str(node.t) == date:
            node.process_parameters[variable]['constant'] = value
    return new_end_timestep

# def wrapper_impervioussurface_sewer(self):
#     def push_to_sewers(#self
#                        ):
#         """Outflow function that distributes ponded water (i.e., surface runoff)
#         to the parent node's attached sewers.

#         Returns:
#             (tuple): A tuple containing a VQIP amount for model inputs and outputs
#                 for mass balance checking.

#         """
#         # Get runoff
#         surface_runoff = self.pull_ponded()
#         print(surface_runoff['volume'])
#         # Distribute
#         # TODO in cwsd_partition this is done with timearea
#         reply = self.parent.push_distributed(surface_runoff, of_type=["Sewer", "SewerCSO"], tag = 'Land')

#         # Update tank (forcing, because if the water can't go to the sewer, where else can it go)
#         _ = self.push_storage(reply, force=True)
#         # TODO... possibly this could flow to attached river or land nodes.. or other surfaces? I expect this doesn't matter for large scale models.. but may need to be revisited for detailed sewer models

#         # Return empty mass balance because outflows are handled by parent
#         return (self.empty_vqip(), self.empty_vqip())
#     return push_to_sewers
# def wrapper_stormsewer_cso(self):
#     def make_discharge(#self
#                        ):
#         """Function to trigger downstream sewer flow.

#         Updates sewer tank travel time, pushes to WWTW, then sewer, then CSO. May
#         flood land if, after these attempts, the sewer tank storage is above
#         capacity.

#         """
#         self.sewer_tank.internal_arc.update_queue(direction="push")
#         # TODO... do I need to do anything with this backflow... does it ever happen?
        
#         # Discharge to CSO first
#         capacity = 86400 * 2 # [m3/d]
        
#         cso_spill = self.sewer_tank.active_storage['volume'] - capacity # [m3/d]
#         if cso_spill > 0:
#             # cso_spill = self.v_change_vqip(self.sewer_tank.active_storage, cso_spill) # [vqip]
#             cso_spill = self.sewer_tank.pull_storage({'volume': cso_spill}) # [vqip]
            
#             remaining = self.push_distributed(cso_spill,
#                                             of_type = ["CSO"]
#                                             )
#             _ = self.sewer_tank.push_storage(remaining, force=True)
#             # # list of arcs that connect with gw bodies
#             # _, arcs = self.get_direction_arcs(direction='push', of_type=['CSO']) # new type
#             # cso_arcs = [arc for arc in arcs if 'cso' in arc.name]
#             # # if there is only one river arc
#             # if len(cso_arcs) == 1:
#             #     arc = cso_arcs[0]
#             #     remaining = arc.send_push_request(cso_spill)
#             #     _ = self.sewer_tank.push_storage(remaining, force = True)
#             #     if remaining['volume'] > constants.FLOAT_ACCURACY:
#             #         print('Sewer unable to push from '+self.name+' into cso '+arc.name.split('-to-')[-1])
#             # else:
#             #     print("More than 1 cso corresponds with "+self.name+" - can't model it at this stage and needs further development")
#         # Discharge to sewer and river then (based on preferences)
#         to_send = self.sewer_tank.pull_storage(self.sewer_tank.active_storage) # [vqip]
#         remaining = self.push_distributed(to_send,
#                                         of_type = ["Sewer", "River_h"]
#                                         )
#         _ = self.sewer_tank.push_storage(remaining, force=True)
#         # #Discharge to WWTW if possible
#         # remaining = self.push_distributed(remaining,
#         #                                 of_type = ["WWTW"],
#         #                                 tag = 'Sewer'
#         #                                 )

#         # remaining = self.push_distributed(self.sewer_tank.active_storage)

#         # TODO backflow can cause mass balance errors here

#         # # Update tank
#         # sent = self.extract_vqip(self.sewer_tank.active_storage, remaining)
#         # reply = self.sewer_tank.pull_storage_exact(sent)
#         # if (reply["volume"] - sent["volume"]) > constants.FLOAT_ACCURACY:
#         #     print("Miscalculated tank storage in discharge")

#         # Flood excess
#         ponded = self.sewer_tank.pull_ponded()
#         if ponded["volume"] > constants.FLOAT_ACCURACY:
#             reply_ = self.push_distributed(ponded, of_type=["Land"], tag="Sewer")
#             reply_ = self.sewer_tank.push_storage(reply_, time=0, force=True)
#             if reply_["volume"]:
#                 print("ponded water cant reenter")
#     return make_discharge


# def wrapper_run(self):
    
#     def run(#self, 
#             dates = None,
#             settings = None,
#             record_arcs = None,
#             record_tanks = None,
#             verbose = True,
#             record_all = True,
#             other_attr = {}):
#         """Run the model object with the default orchestration
    
#         Args:
#             dates (list, optional): Dates to simulate. Defaults to None, which
#                 simulates all dates that the model has data for.
#             settings (dict, optional): Dict to specify what results are stored,
#                 not currently used. Defaults to None.
#             record_arcs (list, optional): List of arcs to store result for. 
#                 Defaults to None.
#             record_tanks (list, optional): List of nodes with water stores to 
#                 store results for. Defaults to None.
#             verbose (bool, optional): Prints updates on simulation if true. 
#                 Defaults to True.
#             record_all (bool, optional): Specifies to store all results.
#                 Defaults to True.
#             other_attr (dict, optional): Dict to store additional attributes of 
#                 specified nodes/arcs. Example: 
#                 {'arc.name1': ['arc.attribute1'],
#                  'arc.name2': ['arc.attribute1', 'arc.attribute2'],
#                  'node.name1': ['node.attribute1'],
#                  'node.name2': ['node.attribute1', 'node.attribute2']}
#                 Defaults to None.
    
#         Returns:
#             flows: simulated flows in a list of dicts
#             tanks: simulated tanks storages in a list of dicts
#             node_mb: mass balance differences in a list of dicts (not currently used)
#             surfaces: simulated surface storages of land nodes in a list of dicts
#             requested_attr: timeseries of attributes of specified nodes/arcs requested by the users
#         """
        
#         if record_arcs is None:
#             record_arcs = self.arcs.keys()
            
#         if record_tanks is None:
#             record_tanks = []
        
#         if settings is None:
#             settings = self.default_settings()
            
#         def blockPrint():
#             stdout = sys.stdout
#             sys.stdout = open(os.devnull, 'w')
#             return stdout
#         def enablePrint(stdout):
#             sys.stdout = stdout
#         if not verbose:
#             stdout = blockPrint() 
#         if dates is None:
#             dates = self.dates
        
#         flows = []
#         tanks = []
#         node_mb = []
#         surfaces = []
        
#         # print(
#         #       self.nodes['6aead97c-0040-4e31-889d-01f628faf990'].data_input_dict)#[('nhx-dry', dates[-36].to_period('M'))])
              
#         for date in tqdm(dates, disable = (not verbose)):
#         # for date in dates:
#             for node in self.nodelist:
#                 node.t = date
#                 node.monthyear = date.to_period('M')
            
#             # print(type(node.monthyear))
            
#             #Run FWTW
#             for node in self.nodes_type['FWTW'].values():
#                 node.treat_water()
            
#             #Create demand (gets pushed to sewers)
#             for node in self.nodes_type['Demand'].values():
#                 node.create_demand()
            
#             #Create runoff (impervious gets pushed to sewers, pervious to groundwater)
#             for node in self.nodes_type['Land'].values():
#                 node.run()
            
#             #Infiltrate GW
#             for node in self.nodes_type['Groundwater'].values():
#                 node.infiltrate()
#             for node in self.nodes_type['Groundwater_h'].values():
#                 node.infiltrate()
            
#             #Foul second so that it can discharge any misconnection
#             for node in self.nodes_type['Foul'].values():
#                 node.make_discharge()
            
#             #Discharge sewers (pushed to other sewers or WWTW)
#             for node in self.nodes_type['Sewer'].values():
#                 node.make_discharge()
            
#             #Discharge WWTW
#             for node in self.nodes_type['WWTW'].values():
#                 node.calculate_discharge()
            
#             #Discharge GW
#             for node in self.nodes_type['Groundwater'].values():
#                 node.distribute()
            
#             #river
#             # for node in self.nodes_type['Lake'].values():
#             #     node.calculate_discharge()
#             for node in self.nodes_type['River'].values():
#                 node.calculate_discharge()
            
#             #Abstract
#             for node in self.nodes_type['Reservoir'].values():
#                 node.make_abstractions()
            
#             for node in self.nodes_type['Land'].values():
#                 node.apply_irrigation()
    
#             for node in self.nodes_type['WWTW'].values():    
#                 node.make_discharge()
            
#             #Catchment routing
#             for node in self.nodes_type['Catchment'].values():
#                 node.route()
            
#             #river
#             # for node in self.nodes_type['Lake'].values():
#             #     node.distribute()
#             for node_name in self.river_discharge_order:
#                 self.nodes[node_name].distribute()
            
            
#             #mass balance checking
#             #nodes/system
#             sys_in = self.empty_vqip()
#             sys_out = self.empty_vqip()
#             sys_ds = self.empty_vqip()
            
#             #arcs
#             for arc in self.arcs.values():            
#                 in_, ds_, out_ = arc.arc_mass_balance()
#                 for v in constants.ADDITIVE_POLLUTANTS + ['volume']:
#                     sys_in[v] += in_[v]
#                     sys_out[v] += out_[v]
#                     sys_ds[v] += ds_[v]
#             for node in self.nodelist:
#                 # print(node.name)
#                 in_, ds_, out_ = node.node_mass_balance()
                
#                 # temp = {'name' : node.name,
#                 #         'time' : date}
#                 # for lab, dict_ in zip(['in','ds','out'], [in_, ds_, out_]):
#                 #     for key, value in dict_.items():
#                 #         temp[(lab, key)] = value
#                 # node_mb.append(temp)
                
#                 for v in constants.ADDITIVE_POLLUTANTS + ['volume']:
#                     sys_in[v] += in_[v]
#                     sys_out[v] += out_[v]
#                     sys_ds[v] += ds_[v]
        
#             for v in constants.ADDITIVE_POLLUTANTS + ['volume']:
                
#                 #Find the largest value of in_, out_, ds_
#                 largest = max(sys_in[v], sys_in[v], sys_in[v])
    
#                 if largest > constants.FLOAT_ACCURACY:
#                     #Convert perform comparison in a magnitude to match the largest value
#                     magnitude = 10**int(log10(largest))
#                     in_10 = sys_in[v] / magnitude
#                     out_10 = sys_in[v] / magnitude
#                     ds_10 = sys_in[v] / magnitude
#                 else:
#                     in_10 = sys_in[v]
#                     ds_10 = sys_in[v]
#                     out_10 = sys_in[v]
                
#                 if (in_10 - ds_10 - out_10) > constants.FLOAT_ACCURACY:
#                     print("system mass balance error for " + v + " of " + str(sys_in[v] - sys_ds[v] - sys_out[v]))
            
#             #Store results
#             for arc in record_arcs:
#                 arc = self.arcs[arc]
#                 flows.append({'arc' : arc.name,
#                               'flow' : arc.vqip_out['volume'],
#                               'time' : date})
#                 for pol in constants.POLLUTANTS:
#                     flows[-1][pol] = arc.vqip_out[pol]
            
#             for node in record_tanks:
#                 node = self.nodes[node]
#                 tanks.append({'node' : node.name,
#                               'storage' : node.tank.storage['volume'],
#                               'time' : date})
#             if record_all:
#                 for node in self.nodes.values():
#                     for prop_ in dir(node):
#                         prop = node.__getattribute__(prop_)
#                         if prop.__class__ in [QueueTank, Tank, ResidenceTank]:
#                             tanks.append({'node' : node.name,
#                                           'time' : date,
#                                           'storage' : prop.storage['volume'],
#                                           'prop' : prop_})
#                             for pol in constants.POLLUTANTS:
#                                 tanks[-1][pol] = prop.storage[pol]
                                    
#                 for name, node in self.nodes_type['Land'].items():
#                     for surface in node.surfaces:
#                         if not isinstance(surface,ImperviousSurface):
#                             surfaces.append({'node' : name,
#                                               'surface' : surface.surface,
#                                               'percolation' : surface.percolation['volume'],
#                                               'subsurface_r' : surface.subsurface_flow['volume'],
#                                               'surface_r' : surface.infiltration_excess['volume'],
#                                               'storage' : surface.storage['volume'],
#                                               'evaporation' : surface.evaporation['volume'],
#                                               'precipitation' : surface.precipitation['volume'],
#                                               'tank_recharge' : surface.tank_recharge,
#                                               'capacity' : surface.capacity,
#                                               'time' : date,
#                                               'et0_coef' : surface.et0_coefficient,
#                                               # 'crop_factor' : surface.crop_factor
#                                               })
#                             for pol in constants.POLLUTANTS:
#                                 surfaces[-1][pol] = surface.storage[pol]
#                         else:
#                             surfaces.append({'node' : name,
#                                               'surface' : surface.surface,
#                                               'storage' : surface.storage['volume'],
#                                               'evaporation' : surface.evaporation['volume'],
#                                               'precipitation' : surface.precipitation['volume'],
#                                               'capacity' : surface.capacity,
#                                               'time' : date})
#                             for pol in constants.POLLUTANTS:
#                                 surfaces[-1][pol] = surface.storage[pol]
            
#             for node in self.nodes.values():
#                 node.end_timestep()
            
#             for arc in self.arcs.values():
#                 arc.end_timestep()
#         if not verbose:
#             enablePrint(stdout)
#         return flows, tanks, node_mb, surfaces
#     return run
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
import math
import copy
import warnings
import cmath
import inspect
import six 

from itertools import chain

from pykat import finesse
from pykat.finesse import BlockedKatFile
from pykat.ifo import *

import pykat.components
import pykat.exceptions as pkex
import pykat.external.peakdetect as peak
import pkg_resources

from scipy.constants import c as clight
from scipy.optimize import fmin

from pykat.optics.hellovinet import hellovinet
from pykat.tools.lensmaker import lensmaker
from pykat.tools.compound_lens import combine

# Changes, Andreas Freise
# - removed unused frequencies, added ports for common DC power and steady locks
# - removed LIGO specific functions (e.g. remove HAM/IMC, suspend pitch)
# - changed DOFs according to VIR-0445A-20
# - fixed: path length for lx and ly seemed swapped, lx should be N and ly should be W
# - fixed: changed adjust_PRC_length and added adjust_SRC_length
# - fixed: changed masses for PR/SR, (todo: zmech TF for main mirrors to suspend_mirrors)
# - fixed: corrected wrong apply_lock_feedback()
# - fixed: in pretune, add zeroing of mod frequencies, made MICH pretune relative

# Todo
# - probably make this file just 'AdV' and generate O3b and O4 configurations
# - allow easy switch between different detector configurations
# - check HOM functions
# - check and clean cold/warm features

class ADVPLUS_IFO(IFO):   
    """
    This contains Advanced Virgo Plus specific methods for computing interferometer
    variables.
    
    Functions that operate on the kat/IFO objects, manipulate their
    structure and return a new one, should not be included here. They should
    be separate functions that are called by the user. 
    
    The functions here should be those that update the kat with information
    from the IFO object or vice-versa.
    """

    def __init__(self, kat, tuning_keys_list, tunings_components_list):
        IFO.__init__(self, kat, tuning_keys_list, tunings_components_list)
        self._f1 = np.nan
        self._f2 = np.nan
        self._f3 = np.nan
        self._etalonN = 0
        self._etalonW = 0
        
        # TODO check if these are needed here or can be added by special functions
        self._f1_3 = np.nan
        self._f2_3 = np.nan
    
    @property
    def DARMoffset(self):
        if 'DCoffset' not in self.kat.data:
            return 0
        elif 'DARM' not in self.kat.data['DCoffset']:
            return 0
        else:
            return float(self.kat.data['DCoffset']['DARM'])

    @DARMoffset.setter
    def DARMoffset(self, value):
        self.set_DC_offset(DCoffset=value, offset_type = 'DARM', verbose=False)

    @property
    def MICHoffset(self):
        if 'DCoffset' not in self.kat.data:
            return 0
        elif 'MICH' not in self.kat.data['DCoffset']:
            return 0
        else:
            return float(self.kat.data['DCoffset']['MICH'])

    @MICHoffset.setter
    def MICHoffset(self, value):
        self.set_DC_offset(DCoffset=value, offset_type = 'MICH', verbose=False)
    
    @property
    def DCoffset(self):
        if 'DCoffset' not in self.kat.data:
            return None
        else:
            return self.kat.data['DCoffset']
    
    @property
    def DCoffsetW(self):
        if 'DCoffsetW' not in self.kat.data:
            return 0
        else:
            return float(self.kat.data['DCoffsetW'])

    @DCoffsetW.setter
    def DCoffsetW(self, value):
        self.kat.data['DCoffsetW'] = float(value)

    @property
    def mirror_properties(self):
        if 'mirror_properties' not in self.kat.data:
            return None
        else:
            return self.kat.data['mirror_properties']
        
    @mirror_properties.setter
    def mirror_properties(self, value):
        self.kat.data['mirror_properties'] = value

    @property
    def cold_optics(self):
        if 'cold_optics_parameters' in self.kat.data:
            return self.kat.data['cold_optics_parameters']
        else:
            return None

    @cold_optics.setter
    def cold_optics(self, value):
        self.kat.data['cold_optics_parameters'] = value    

    
    @property
    def etalonW(self):
        return self._etalonW
    @etalonW.setter
    def etalonW(self, value):
        self._etalonW=value

    @property
    def etalonN(self):
        return self._etalonN
    @etalonN.setter
    def etalonN(self, value):
        self._etalonN=value

    @property
    def f1(self):
        return self._f1
    @f1.setter
    def f1(self, value):
        self._f1 = float(value)
        self.f1_3 = 3.0 * self.f1 
        # Updating ports
        if hasattr(self, 'LSC_DOFs'):
            for a in self.LSC_DOFs:
                if a.port.name[-2:] == 'f1':
                    a.port.f = self.f1
                
    @property
    def f2(self):
        return self._f2
    @f2.setter
    def f2(self, value):
        self._f2 = float(value)
        self.f2_3 = 3.0 * self.f2 
        # Updating ports
        if hasattr(self, 'LSC_DOFs'):
            for a in self.LSC_DOFs:
                if a.port.name[-2:] == 'f2':
                    a.port.f = self.f2
        
    @property
    def f3(self):
        return self._f3
    @f3.setter
    def f3(self, value):
        self._f3 = float(value)
        # Updating ports
        if hasattr(self, 'LSC_DOFs'):
            for a in self.LSC_DOFs:
                if a.port.name[-2:] == 'f3':
                    a.port.f = self.f3
        
    def createPorts(self):
        # useful ports
        # TODO phases should probably be zero?
        # TODO nB1p does not yet exist
        self.B1   = Output(self, "B1", "nB1")
        self.B1p   = Output(self, "B1p", "nB1p")
        self.B2   = Output(self, "B2", "nB2")
        self.B4   = Output(self, "B4", "nB4")
        self.B5   = Output(self, "B5", "nBSsn*")
        self.B7   = Output(self, "B7", "nB7")
        self.B8   = Output(self, "B8", "nB8")
        self.POW_BS  = Output(self, "PowBS", "nBSs*")
        self.POW_N   = Output(self, "PowN",  "nNI2")
        self.POW_W   = Output(self, "PowW",  "nWI2")

        self.B1p_f2 = Output(self, "B1p_f2", "nB1p", self.f2, phase=101)
        self.B2_f1 = Output(self, "B2_f1", "nB2", self.f1, phase=101)
        self.B2_f2 = Output(self, "B2_f2", "nB2", self.f2, phase=14)
        self.B2_f3 = Output(self, "B2_f3", "nB2", self.f3, phase=14)
        self.B4_f1  = Output(self, "B4_f1",  "nB4",  self.f1, phase=101)
        self.B4_f2  = Output(self, "B4_f2",  "nB4",  self.f2, phase=13)
    
    def update_LSC_DOFs(self):
        self.LSC_DOFs = (self.PRCL, self.MICH, self.CARM, self.DARM)
        if self.isSRC:
            self.LSC_DOFs = self.LSC_DOFs + (self.SRCL,)
        self.update()

    def compute_derived_resonances(self):
        self.fsrX = 0.5 * clight / float(self.kat.LN.L)
        self.fsrY = 0.5 * clight / float(self.kat.LW.L)
    
        # definition of f1, TDR section 2.3
        self.f1_arm = 125.5 * self.fsrX - 300.0 

        self.fsrPRC = 0.5 * clight / self.lPRC
        #self.fsrSRC = 0.5 * clight / self.lSRC
        self.fsrSRC = None
        self.f1_SRC = None
        if self.isSRC:
            self.fsrSRC = 0.5 * clight / self.lSRC
            self.f1_SRC = 3.5 * self.fsrSRC

        self.f1_PRC = 3.5 * self.fsrPRC
    
    def compute_derived_lengths(self, verbose=False):
        # Optical path lengths
        ######################################
        # Optical path length from PR HR to BS HR
        self.lpr = self.kat.lPR_POP.L + self.kat.sPOPsub.L*self.kat.sPOPsub.n + self.kat.lPOP_BS.L
        # Optical path length from BS HR to NI HR
        self.lx = (self.kat.sBSsub1.L * self.kat.sBSsub1.n + self.kat.lBS_CPN.L +
                   self.kat.sCPNsub.L * self.kat.sCPNsub.n + self.kat.sCPN_NI.L +
                   self.kat.sNIsub.L * self.kat.sNIsub.n )
        # Optical path length from BS HR to WI HR
        self.ly = (self.kat.lBS_CPW.L + self.kat.sCPWsub.L * self.kat.sCPWsub.n +
                   self.kat.sCPW_WI.L + self.kat.sWIsub.L * self.kat.sWIsub.n)

        # resulting combined distances (single, not roundtrip)
        self.lMI =  0.5 * (self.lx + self.ly)
        self.lPRC = self.lpr + self.lMI
        self.lSchnupp = self.lx - self.ly
        self.lsr = None
        self.lSRC = None
        if self.isSRC:
            self.lsr = self.kat.lsr.L + self.kat.sBSsub2.L*self.kat.sBSsub2.n
            self.lSRC = self.lsr + self.lMI


        # Effective geometric distances (for mode matching)
        ######################################################
        ## From PR HR to BS HR
        #self.lpr_g = self.kat.lPR_POP.L + self.kat.sPOPsub.L.value/self.kat.sPOPsub.n.value + self.kat.lPOP_BS.L
        ## Optical path length from BS HR to NI HR
        #self.lx_g = (self.kat.sBSsub1.L.value/self.kat.sBSsub1.n.value + self.kat.lBS_CPN.L +
        #             self.kat.sCPNsub.L.value/self.kat.sCPNsub.n.value + self.kat.sCPN_NI.L +
        #             self.kat.sNIsub.L.value/self.kat.sNIsub.n.value )
        ## From BS HR to WI HR
        #self.ly_g = (self.kat.lBS_CPW.L + self.kat.sCPWsub.L.value/self.kat.sCPWsub.n.value +
        #             self.kat.sCPW_WI.L + self.kat.sWIsub.L.value/self.kat.sWIsub.n.value)
        #self.lsr_g = None
        #if self.isSRC:
        #    pass
        ## resulting combined distances (single, not roundtrip)
        #self.lMI_g =  0.5 * (self.lx_g + self.ly_g)
        #self.lPRC_g = self.lpr_g + self.lMI_g
        #self.lSRC_g = None
        #if self.isSRC:
        #    self.lSRC_g = self.lsr_g + self.lMI_g

        #self.lSchnupp_g = self.lx_g - self.ly_g
        
        self.compute_derived_resonances()
    
    def suspend_mirrors_z(self):
        """
        Suspends the main mirrors in an AdV+ model in z in the
        supplied kat object.
    
        Returns the commands used for reference.
        """
        
        # TODO get numbers for mirror masses!
        # TODO get reference for mech TF
        # main mirror masses in TDR, mirror dimensions for BS and PR/SR in TDR
        # Paolo Ruggi 28.09.2020
        # NE
        # zero: 0.653 Hz Q = 80
        # pole: 0.419 Hz, Q = 100
        # pole: 0.760 Hz, Q = 20

        # WE
        # zero: 0.653 Hz Q = 80
        # pole: 0.416 Hz, Q = 100
        # pole: 0.755 Hz, Q = 20

        # add suspension TF later again
        #tf suspension 1 0 z 0.7 4200 p 0.4 2578 p 0.8 5000
        #attr WE M 42 zmech suspension
        code = """
        attr WE M 42 
        attr NE M 42 
        attr WI M 42 
        attr NI M 42 
        attr PR M 21
        attr BS M 34
        """
        if self.isSRC:
            code += "attr SR M 21"
        self.kat.parse(code)
        return code
    
    def fix_mirrors(self, z=True, pitch=True, yaw=True):
        """
        This function will iterate through the mirrors
        and remove any suspension settings on them. This can be
        done individuallly or for z, pitch, and yaw.
        """
    
        for m in self.mirrors:
            if m is not None:
                mirror = self.kat.components[m]
                if z:
                    mirror.mass = None
                    mirror.zmech = None
                if pitch:
                    mirror.Iy = None
                    mirror.rymech = None
                if yaw:
                    mirror.Ix = None
                    mirror.rxmech = None
        
    def lengths_status(self):
        self.compute_derived_lengths()
        print(" .--------------------------------------------------.")
        print("| - Arm lengths [m]:                                |")
        print("| Ln   = {:<11.4f} Lw  = {:<11.4f}              |".format(float(self.kat.LN.L), float(self.kat.LW.L)))
        print("| - Michelson and recycling lengths [m]:            | ")
        print("| ln   = {:<11.4f} lw  = {:<11.4f}              |".format(self.lx, self.ly))
        if self.lsr is None:
            print("| lpr  = {:<11.4f} {:20}           |".format(self.lpr, ""))
        else:
            print("| lpr  = {:<11.4f} lsr  = {:<11.4f}             |".format(self.lpr, self.lsr))

        print("| lMI  = {:<11.4f} lSchnupp = {:<11.4f}         |".format(self.lMI, self.lSchnupp))
        if self.lSRC is None:
            print("| lPRC = {:<11.4f} {:20}           |".format(self.lPRC, ""))
        else:
            print("| lPRC = {:<11.4f} lSRC = {:<11.4f}             |".format(self.lPRC, self.lSRC))
        print("+---------------------------------------------------+")
        print("| - Associated cavity frequencies [Hz]:             |")
        print("| fsrx   = {:<11.2f},    fsry = {:<11.2f}       |".format(self.fsrX, self.fsrY))
        if self.fsrSRC is None:
            print("| fsrPRC = {:<11.2f} {:19}       |".format(self.fsrPRC, ""))
        else:
            print("| fsrPRC = {:<11.2f}, fsrSRC = {:<11.2f}        |".format(self.fsrPRC, self.fsrSRC))
        # print("| f1_PRC = {:11.8}                             |".format(self.f1_PRC))
        print("| - Modulation sideband frequencies [MHz]:          |")
        print("| f1     = {:<12.6f},   f2   = {:<12.6f}      |".format(self.f1/1e6, self.f2/1e6))
        print("| f3     = {:<12.6f}                             |".format(self.f3/1e6))

        print(" +--------------------------------------------------'")
        print("| - Check frequency match:                          |")
        print("| 125.5*fsrx-300 = {:<8.6f} MHz                     |".format((125.5*self.fsrX-300)/1e6))
        print("| 0.5*fsrPRC = {:<8.6f} MHz                         |".format(0.5*self.fsrPRC/1e6))
        print("| 0.5*fsrSRC = {:<8.6f} MHz                         |".format(0.5*self.fsrSRC/1e6))
        print("| 9*f1 = {:<9.6f} MHz                              |".format(9*self.f1/1e6))
        print(" `--------------------------------------------------'")
    
    def remove_modulators(self):
        """
        Removes the input modulators and reconnects the input laser to the PRC reflection node.
        
        This function alters the kat object directly.
        """
        self.kat.s1.L = (self.kat.s1.L + self.kat.s0.L.value +
                         self.kat.sEOM1.L.value + self.kat.sEOM2.L.value)
        
        self.kat.remove("s0", "EOM1", "sEOM1", "EOM2", "sEOM2", 'EOM3') # Remove modulators
        # Set output node of laser block to be on the laser
        # TODO test this
        self.kat.nodes.replaceNode(self.kat.s1, 'nEOM3b', 'nin')
        #self.kat.nodes.replaceNode(self.kat.i1, 'nin', 'nEOM3b')

    def remove_OMC(self):
        """
        Method for removing the OMC and OMCpath blocks. The SR mirror
        is reconnected to the sOut space, so that nB1 remains a valid
        output node.
        """

        # TODO check handling of cold optics
        # Removing from cold optics. Perhaps change to never include these in the
        # first place.
        #tmpCold = copy.copy(self.cold_optics)
        #for k,v in tmpCold.items():
        #    if k[:3] == 'OMC' or k[:3] == 'MMT':
        #        self.cold_optics.pop(k)

        self.kat.removeBlock('OMCpath')
        # connect MMT to sout
        #self.kat.nodes.replaceNode(self.kat.sOut,self.kat.sOut.nodes[0],self.kat.components['MMT_L2'].nodes[1].name)

        self.kat.removeBlock('OMC')
        
        # connect SR output note to sOut
        self.kat.nodes.replaceNode(self.kat.sOut,self.kat.sOut.nodes[0],self.kat.components['SRAR'].nodes[1].name)


        
    def adjust_PRC_length(self, verbose=False):
        """
        Adjust PRC length so that it fulfils the requirement
        lPRC = 0.5 * c / (2 * f1), see TDR 2.3
    
        This function directly alters the lengths of the associated kat object.
        """
        kat = self.kat
        
        # TODO check if we want to do verbose OR katverbose instead
        vprint(verbose, "-- adjusting PRC length")
        ltmp = 0.5 * clight / (2 * kat.IFO.f1)
        delta_l = ltmp - kat.IFO.lPRC
        vprint(verbose, "   adusting kat.lPOP_B5.L by {:.4g}m".format(delta_l))
        kat.lPOP_BS.L += delta_l
    
        kat.IFO.compute_derived_lengths(kat)

    def adjust_SRC_length(self, verbose=False):
        """
        Adjust SRC length so that it fulfils the requirement
        lSRC = 0.5 * c / (2 * f1), see TDR 2.3
    
        This function directly alters the lengths of the associated kat object.
        """
        kat = self.kat
        
        vprint(verbose, "-- adjusting SRC length")
        ltmp = 0.5 * clight / (2 * kat.IFO.f1)
        delta_l = ltmp - kat.IFO.lSRC
        vprint(verbose, "   adusting kat.lsr.L by {:.4g}m".format(delta_l))
        kat.lsr.L += delta_l
    
        kat.IFO.compute_derived_lengths(kat)

    def apply_lock_feedback(self, out, idx=None):
        """
        This function will apply the lock values that have been calculated
        in a previous kat run. This should bring the kat object closer to an
        initial lock point so that the lock commands do not need to be run
        on startup.
        
        out: kat run object containing data on lock outputs
        idx: the step in the output array to use
        
        This function directly alters the tunings of the associated kat object.

        TODO: don't use hard-coded mirror names but use DOF info.
        """
        
        tuning = self.kat.IFO.get_tunings()
    
        if "NE_lock" in out.ylabels:
            if idx is None:
                tuning["NE"] += float(out["NE_lock"])
            else:
                tuning["NE"] += float(out["NE_lock"][idx])
        else:
            pkex.printWarning("could not find NE lock")
        
        if "WE_lock" in out.ylabels:
            if idx is None:
                tuning["WE"] += float(out["WE_lock"])
            else:
                tuning["WE"] += float(out["WE_lock"][idx])
        else:
            pkex.printWarning("could not find WE lock")
        
        if "PRCL_lock" in out.ylabels:
            if idx is None:
                tuning["PR"]  += float(out["PR_lock"])
            else:
                tuning["PR"]  += float(out["PR_lock"][idx])
        else:
            pkex.printWarning("could not find PR lock")
        
        if ("WI_lock" in out.ylabels):
            if idx is None:
                tuning["WI"] += float(out["WI_lock"])
            else:
                tuning["WI"] += float(out["WI_lock"][idx])
        else:
            pkex.printWarning("could not find WI lock")
        if ("NI_lock" in out.ylabels):
            if idx is None:
                tuning["NI"] += float(out["NI_lock"])
            else:
                tuning["NI"] += float(out["NI_lock"][idx])
        else:
            pkex.printWarning("could not find NI lock")
        
        if "SR_lock" in out.ylabels:
            if idx is None:
                tuning["SR"]  += float(out["SR_lock"])
            else:
                tuning["SR"]  += float(out["SR_lock"][idx])
        else:
            pkex.printWarning("could not find SR lock")
         
        self.kat.IFO.apply_tunings(tuning)
    
    def set_DC_offset(self, DCoffset=None, offset_type = 'DARM', verbose=False):
        """
        Sets the DC offset for the interferometer. It can be set to DARM or MICH. 
        This function directly alters the tunings of the associated kat object.
        If no DCoffset is specified, the function finds the DC offset that yields
        5 times the current dark port power.

        Parameters
        ----------
        DCoffset     - Offset to apply to the chosen degree of freedom [degrees]
        offset_type  - String specifying the degree of freedom to apply the DC
                       offset to. Must be DARM or MICH.
        """

        # TODO allow different options, for example absolute DF power of 4mW
        if not "DCoffset" in self.kat.data:
            self.kat.data['DCoffset'] = {}

        # Checking if DARM or MICH is used
        if offset_type == 'DARM' or offset_type == 'darm':
            isDARM = True
            if self.DARMoffset != 0:
                pkex.printWarning(("WARNING! A DARM offset is alredy set. The function"+
                                   "set_DC_offset() overwrites previous DARM-offset in "+
                                   "kat.IFO.DARMoffset, but the tunings might bee added "+
                                   "in the kat-object. Make sure to only add offset once "+
                                   "if thermal functions are to be used, or run pretune() "+
                                   "in between to reset the offset."))
        elif offset_type == 'MICH' or offset_type == 'mich':
            isDARM = False
            if self.MICHoffset != 0:
                pkex.printWarning(("WARNING! A MICH offset is alredy set. The function"+
                                   "set_DC_offset() overwrites previous MICH-offset in "+
                                   "kat.IFO.MICHoffset, but the tunings might bee added "+
                                   "in the kat-object. Make sure to only add offset once "+
                                   "if thermal functions are to be used, or run pretune() "+
                                   "in between to reset the offset."))
        else:
            raise pkex.BasePyKatException("\033[91m offset_type must be DARM or MICH. \033[0m")

        vprint(verbose, "-- applying user-defined DC offset to {}:".format(offset_type))

        _kat = self.kat

        if DCoffset:
            tunings = self.get_tunings()
            if isDARM:
                for name, factor in zip(self.DARM.optics, self.DARM.factors):
                    tunings[name] += DCoffset*factor 
                self.kat.data['DCoffset']['DARM'] = DCoffset

            else:
                self.kat.data['DCoffset']['MICH'] = DCoffset
                for name, factor in zip(self.MICH.optics, self.MICH.factors):
                    tunings[name] += DCoffset*factor 
                
            self.apply_tunings(tunings)
            
            # Compute the DC offset powers
            kat = _kat.deepcopy()
        
            signame = kat.IFO.B1.add_signal()
        
            kat.noxaxis=True
            kat.yaxis = 'lin abs'
            out = kat.run(cmd_args=["-cr=on"])
        
            self.DCoffsetW = float(out[signame])
        else:
            # Finding light power in AS port (mostly due to RF sidebands now)
            kat = _kat.deepcopy()
        
            signame = kat.IFO.B1.add_signal()
        
            kat.noxaxis=True
        
            out = kat.run()
        
            vprint(verbose, "-- adjusting {} DCoffset based on light in dark port:".format(offset_type))
        
            waste_light = round(float(out[signame]),1)
            vprint(verbose, "   waste light in AS port of {:2} W".format(waste_light))
            
            if waste_light<1e-4:
                waste_light=1e-4

            #kat_lock = _kat.deepcopy()
        
            DCoffset = self.find_DC_offset(5*waste_light, offset_type, verbose=verbose)
            
        vprint(verbose, "   {} DCoffset = {:6.4} deg ({:6.4} m)".format(offset_type, DCoffset,
                                                                        DCoffset / 360.0 * _kat.lambda0))
        vprint(verbose, "   at dark port power: {:6.4} W".format(self.DCoffsetW))

    def find_DC_offset(self, AS_power, offset_type = 'DARM', precision=1e-6, verbose=False):
        """
        Returns the DC offset of DARM or MICH that corresponds to the specified power in the B1 power.
        
        This function directly alters the tunings of the associated kat object.
        """

        if offset_type == 'DARM' or offset_type == 'darm':
            isDARM = True
            if self.DARMoffset != 0:
                pkex.printWarning(("WARNING! A DARM offset is alredy set. The function"+
                                   "set_DC_offset() overwrites previous DARM-offset in "+
                                   "kat.IFO.DARMoffset, but the tunings might bee added "+
                                   "in the kat-object. Make sure to only add offset once "+
                                   "if thermal functions are to be used, or run pretune() "+
                                   "in between to reset the offset."))
        elif offset_type == 'MICH' or offset_type == 'mich':
            isDARM = False
            if self.MICHoffset != 0:
                pkex.printWarning(("WARNING! A MICH offset is alredy set. The function"+
                                   "set_DC_offset() overwrites previous MICH-offset in "+
                                   "kat.IFO.MICHoffset, but the tunings might bee added "+
                                   "in the kat-object. Make sure to only add offset once "+
                                   "if thermal functions are to be used, or run pretune() "+
                                   "in between to reset the offset."))
        else:
            raise pkex.BasePyKatException("\033[91m offset_type must be DARM or MICH. \033[0m")

        vprint(verbose, "   finding {} DC offset for AS power of {:3g} W".format(offset_type, AS_power))
    
        _kat = self.kat
        
        kat = _kat.deepcopy()
        kat.verbose = False
        kat.noxaxis = True
        
        kat.removeBlock("locks", False)
        kat.removeBlock("errsigs", False)
        
        kat.IFO.B1.add_signal()

        tunings = self.get_tunings()
        
        def powerDiff(phi):
            if isDARM:
                for name, factor in zip(self.DARM.optics, self.DARM.factors):
                    kat.components[name].phi = tunings[name] + phi*factor 
            else:
                for name, factor in zip(self.MICH.optics, self.MICH.factors):
                    kat.components[name].phi = tunings[name] + phi*factor 
                
            out = kat.run()
            # print(verbose, "   ! ", out[self.B1.get_signal_name()], phi)            
            return np.abs(out[self.B1.get_signal_name()] - AS_power)

        vprint(verbose, "   starting peak search...")
        out = fmin(powerDiff, 0, xtol=precision, ftol=1e-3, disp=verbose)
    
        vprint(verbose, "   ... done")
        vprint(verbose, "   DC offset for B1 = {} W is: {:.3e} deg".format(AS_power, out[0]))
        
        tunings = self.get_tunings()

        self.DCoffsetW = AS_power

        if not 'DCoffset' in self.kat.data:
            self.kat.data['DCoffset'] = {}

        if isDARM:
            self.kat.data['DCoffset']['DARM'] = round(out[0], 6)
            DCoffset = self.DARMoffset
            for name, factor in zip(self.DARM.optics, self.DARM.factors):
                tunings[name] += self.DARMoffset*factor 
        else:
            self.kat.data['DCoffset']['MICH'] = round(out[0], 6)
            DCoffset = self.MICHoffset
            for name, factor in zip(self.MICH.optics, self.MICH.factors):
                tunings[name] += self.MICHoffset*factor
            
        self.apply_tunings(tunings)
        
        return DCoffset

    def sensitivity_detector_cmd(self):
        if self.DARM_h.port.f is None:
            cmd = "qnoisedS NSR 1 $fs {node}".format(node=self.DARM_h.port.nodeName[0])
        else:
            cmd = "qnoisedS NSR 2 {f} {phi} $fs {node}".format(f=self.DARM_h.port.f,
                                                               phi=self.DARM_h.port.phase,
                                                               node=self.DARM_h.port.nodeName[0])
        return cmd

    def etalon_cmd(self, dir="N", value=0):
        vname = f"etalon{dir}"
        cmd = f"var {vname} {value}\n"
        cmd += f"set {vname}re {vname} re\n"
        m_name = f"{dir}I"
        ar_name = f"{dir}IAR"
        cmd += f"set {m_name}phi {m_name} phi\n"
        cmd += f"func eta{dir} = ${m_name}phi + (${vname}re)\n"
        cmd += f"noplot eta{dir}\n"
        cmd += f"put {ar_name} phi $eta{dir}\n"
        return cmd

    def add_etalon_block(self, etalonW=0, etalonN=0, verbose=False):
        """
        Creates a lock which links the NIAR and WIAR surfaces to the
        respective mirror HR surface.

        Removes exisiting errsigs block if present.
        
        Returns the commands added for reference.
        """        

        kat = self.kat
        kat.removeBlock("etalon",failOnBlockNotFound=False)

        code1 = self.etalon_cmd("N", etalonN)
        code2 = self.etalon_cmd("W", etalonW)
        kat.parse(code1+code2, addToBlock="etalon")

        return code1+code2    

    def add_errsigs_block(self, noplot=True, verbose=False):
        """
        Creates and adds the 'errsigs' block to the kat object based on the
        DARM, CARM, PRCL, MICH and SRCL DOF objects
        
        Removes exisiting errsigs block if present.
        
        Returns the commands added for reference.
        """
        kat = self.kat
        
        code2 = ""
        for dof in kat.IFO.LSC_DOFs:
            code2 += "\n".join(dof.signal()) + "\n"
        
        code3= ""
    
        # TODO add noplot option back in
        if noplot:
            nameDARM = kat.IFO.DARM.signal_name()
            nameCARM = kat.IFO.CARM.signal_name()
            namePRCL = kat.IFO.PRCL.signal_name()
            nameMICH = kat.IFO.MICH.signal_name()
            if self.isSRC:
                nameSRCL = kat.IFO.SRCL.signal_name()
        
            # code3 = """
            #         noplot {}
            #         noplot {}
            #         noplot {}
            #         noplot {}
            #         noplot {}""".format(nameDARM, nameCARM, namePRCL, nameMICH, nameSRCL).replace("  ","")
                    
        if verbose:
            print(" .--------------------------------------------------")
            print(" | error signals:                                   ")
            print(" +--------------------------------------------------")
            for dof in kat.IFO.LSC_DOFs:
                #for l in code2.splitlines():
                print (" | {:4}: {:44}".format(dof.name, dof.signal()[0]))
            print(" `--------------------------------------------------")

        cmds = "".join([code2, code3])
        kat.removeBlock("errsigs", False)
        kat.parse(cmds, addToBlock="errsigs")
        
        return cmds
        
    def add_locks_block(self, lock_data, verbose=False):
        """
        Accepts a dictionary describing the lock gains and accuracies, e.g.:
            data = {
                "DARM": {"accuracy":1, "gain":1},
                "CARM": {"accuracy":1, "gain":1},
                "PRCL": {"accuracy":1, "gain":1},
                "MICH": {"accuracy":1, "gain":1},
                "SRCL": {"accuracy":1, "gain":1},
            }
        
        This then generates the lock block and adds it to the kat object in the 'locks' block.
        
        Removes exisiting locks block if present.
        
        Returns the commands added for reference.
        """
        
        DOFs = ["DARM", "CARM", "PRCL", "MICH"]
        if self.isSRC:
            DOFs.append('SRCL')
        
        names = [getattr(self, _).signal_name() for _ in DOFs]
        accuracies = [lock_data[_]['accuracy'] for _ in DOFs]
        gains = [lock_data[_]['gain'] for _ in DOFs]

        # Set commands
        code1 = ""
        for dof,name in zip(DOFs,names):
            code1 += "set {}_err {} re\n".format(dof,name)

        # Lock commands
        code2 = ""
        # Noplot commands
        code3 = ""
        for k,dof in enumerate(DOFs):
            if k==0:
                code2 += "lock {0}_lock ${0}_err {1:8.2} {2:8.2} {3}\n".format(dof, gains[k], accuracies[k], -self.kat.IFO.DCoffsetW)
            else:
                code2 += "lock {0}_lock ${0}_err {1:8.2} {2:8.2}\n".format(dof,gains[k],accuracies[k])
            code3 += "noplot {}_lock\n".format(dof)
            
        code4 = ""
        code5 = ""
        for m in self.get_tuning_comps():
            code_tmp = "func {}_lock =".format(m)
            k = 0
            for dof in DOFs:
                if m in self.DOFs[dof].optics:
                    factor = self.DOFs[dof].factors[self.DOFs[dof].optics.index(m)]
                    if k>0:
                        code_tmp += " +"
                    code_tmp += " ({}) * ${}_lock".format(factor, dof)
                    k += 1
            if not code_tmp[-1] == "=":
                code4 += code_tmp + "\n"
                code5 += "put* {0} phi ${0}_lock\n".format(m)
                code3 += "noplot {}_lock\n".format(m)
    
        if verbose:
            print(" .--------------------------------------------------.")
            print(" | Lock commands used:                              |")
            print(" +--------------------------------------------------+")
            for l in code2.splitlines():
                print (" | {:49}|".format(l))
            print(" `--------------------------------------------------'")

        cmds = "".join([code1, code2, code4, code5, code3])
        
        self.kat.removeBlock("locks", False) # Remove existing block if exists
        self.kat.parse(cmds, addToBlock="locks")
        
        return cmds
    
    def add_REFL_gouy_telescope(self, loss=0, gouy_REFL_BS=0, gouy_A=0, gouy_B=90):
        """
        Adds in the gouy phase telescope for WFS detectors and the IFO port objects.
        Commands added into block "REFL_gouy_tele". This attaches to the
        nB2 node which should be from an isolator on the input path.
        
        Also adds the relevant IFO port objects for generating detectors:
            * ASC_REFL9A, ASC_REFL9B
            * ASC_REFL45A, ASC_REFL45B
            * ASC_REFL36A, ASC_REFL36B
        
        These ports are associated with the block "REFL_gouy_tele".
        
        loss: Total loss accumulated along telescope up to the WFS BS [0 -> 1]
        gouy_REFL_BS:  Gouy phase along path from isolator to WFS BS [deg]
        gouy_A: Gouy phase along A path from BS to WFS [deg]
        gouy_B: Gouy phase along B path from BS to WFS [deg]
        """
        
        self.kat.removeBlock("REFL_gouy_tele", False) # Remove old one
        
        self.kat.parse("""
        s  sFI_REFL_WFS_LOSS 0 nB2 nB2_loss1
        m2 mREFL_WFS_loss 0 {} 0 nB2_loss1 nB2_loss2
        s  sFI_REFL_WFS 0 nB2_loss2 nB2_WFS_BS1
        bs WFS_REFL_BS 0.5 0.5 0 0 nB2_WFS_BS1 nB2_WFS_BS2 nB2_WFS_BS3 dump
        s  sWFS_REFL_A  0 nB2_WFS_BS3 nB2_WFS_A
        s  sWFS_REFL_B  0 nB2_WFS_BS2 nB2_WFS_B
        """.format(loss), addToBlock="REFL_gouy_tele", exceptionOnReplace=True)
        
        self.set_REFL_gouy_telescope_phase(gouy_REFL_BS, gouy_A, gouy_B)
        
        self.kat.IFO.ASC_REFL9A   = Output(self.kat.IFO, "ASC_REFL9A",  "nB2_WFS_A",  self.kat.IFO.f1, block="REFL_gouy_tele")
        self.kat.IFO.ASC_REFL9B   = Output(self.kat.IFO, "ASC_REFL9B",  "nB2_WFS_B",  self.kat.IFO.f1, block="REFL_gouy_tele")

        self.kat.IFO.ASC_REFL45A  = Output(self.kat.IFO, "ASC_REFL45A",  "nB2_WFS_A",  self.kat.IFO.f2, block="REFL_gouy_tele")
        self.kat.IFO.ASC_REFL45B  = Output(self.kat.IFO, "ASC_REFL45B",  "nB2_WFS_B",  self.kat.IFO.f2, block="REFL_gouy_tele")
        
        self.kat.IFO.ASC_REFL36A  = Output(self.kat.IFO, "ASC_REFL36A",  "nB2_WFS_A",  self.kat.IFO.f36M, block="REFL_gouy_tele")
        self.kat.IFO.ASC_REFL36B  = Output(self.kat.IFO, "ASC_REFL36B",  "nB2_WFS_B",  self.kat.IFO.f36M, block="REFL_gouy_tele")
        
        self.update()
        
    def set_REFL_gouy_telescope_phase(self, gouy_REFL_BS, gouy_A, gouy_B):
        """
        Sets the gouy phase from the the FI to the REFL WFS BS, and then
        the gouy on each path to the A and B detectors. Units all in degrees.
        """
        
        if "REFL_gouy_tele" in self.kat.getBlocks():
            self.kat.sFI_REFL_WFS.gouy = gouy_REFL_BS
            self.kat.sWFS_REFL_A.gouy = gouy_A
            self.kat.sWFS_REFL_B.gouy = gouy_B
        else:
            raise pkex.BasePyKatException("\033[91mREFL Gouy phase telescope isn't in the kat object, see kat.IFO.add_REFL_gouy_telescope()\033[0m")
        
    def scan_REFL_gouy_telescope_gouy_cmds(self, start, end, steps=20, xaxis=1, AB_gouy_diff=None, relative=False):
        """
        This will return commands to scan the REFL gouy telescope gouy phase of the A and B paths.
        """
        if "REFL_gouy_tele" not in self.kat.getBlocks():
            raise pkex.BasePyKatException("\033[91mREFL Gouy phase telescope isn't in the kat object, see kat.IFO.add_REFL_gouy_telescope()\033[0m")
        
        if xaxis not in [1, 2]:
            raise pkex.BasePyKatException("xaxis value must be 1 or 2")
        elif xaxis == 1:
            xaxis_cmd = "xaxis"
        elif xaxis == 2:
            xaxis_cmd = "x2axis"
            
        if AB_gouy_diff is None:
            AB_gouy_diff = self.kat.sWFS_REFL_B.gouy - self.kat.sWFS_REFL_A.gouy
            
        if relative:
            put = "put*"
        else:
            put = "put"
            
        cmds = ("var REFL_GOUY_SCAN 0\n"
        "{xaxis} REFL_GOUY_SCAN re lin {start} {end} {steps}\n"
        "{put} sWFS_REFL_A gx $x{axis}\n"
        "{put} sWFS_REFL_A gy $x{axis}\n"
        "func REFL_SCAN_B = $x{axis} + {AB_gouy_diff}\n"
        "{put} sWFS_REFL_B gx $REFL_SCAN_B\n"
        "{put} sWFS_REFL_B gy $REFL_SCAN_B\n").format(xaxis=xaxis_cmd, axis=xaxis, start=start, end=end, steps=steps, AB_gouy_diff=AB_gouy_diff, put=put)
        
        return cmds
    
    def update(self):
        """
        Iterates through the IFO and updates the DOFs and Outputs dictionaries with the latest ports and DOFs that have
        been added to the interferometer object.
        """
        self.DOFs = {}
    
        for _ in inspect.getmembers(self, lambda x: isinstance(x, DOF)):
            self.DOFs[_[0]] = _[1]
        
        self.Outputs = {}
    
        for _ in inspect.getmembers(self, lambda x: isinstance(x, Output)):
            self.Outputs[_[0]] = _[1]

    def find_maxtem(self, tol=1e-4, start = 0, stop = 10, verbose=False):
        '''
        Finding the minimum required maxtem for the power to converge to within the relative tolerance tol.
        '''
        kat1 = self.kat.deepcopy()
        sigs = []
        sigs.append(kat1.IFO.POW_BS.add_signal())
        sigs.append(kat1.IFO.POW_X.add_signal())
        sigs.append(kat1.IFO.POW_Y.add_signal())
        if kat1.IFO.isSRC:
            sigs.append(kat1.IFO.B1p.add_signal())

        nsigs = len(sigs)
        kat1.parse('noxaxis\nyaxis abs')
        run = True
        P_old = np.zeros([2, nsigs], dtype=float) + 1e-9
        P = np.zeros(nsigs, dtype=float) + 1e-9
        rdiff = np.ones([2, nsigs], dtype=float)
        mxtm = 0
        while run and mxtm <= stop:
            vprint(verbose, mxtm, ": ")
            P_old[0,:] = P_old[1,:]
            P_old[1,:] = P
            rdiff_old = rdiff[1,:]
            kat1.maxtem = mxtm
            out = kat1.run()
            # print(out.stdout)
            for k,s in enumerate(sigs):
                P[k] = out[s]
            rdiff = np.abs((P-P_old)/P_old)
            if rdiff.max()<tol:
                run = False
            
            vprint(verbose, "{0[0]:.2e} {0[1]:.2e} {0[2]:.2e}".format(rdiff[1,:]))
            # print(kat1.maxtem, rdiff, rdiff.max())
            mxtm += 1
        mxtm -= 1
        if not run:
            # Stepping back to the lowest acceptable maxtem
            mxtm -= 1
            # One more step back if the two previous iterations were within the tolerance
            if rdiff_old.max() < tol:
                mxtm -= 1
        self.kat.maxtem = mxtm
        vprint(verbose, "\nMaxtem set to {}".format(mxtm))


    def find_warm_detector(self, mirror_list, DCoffset=None, tol=1e-5, lensing = True, RoC = True, verbose=False):
        """
        Computes the thermal effects for the mirrors specified in mirror_list and sets the
        warm interferometer values. For an input test masse, the thermal lens is computed
        and is combined with the CP-lens into a new CP-lens.

    
        mirror_list  - List of mirror names to compute the thermal effect for
        DCoffset     - Dictionary with the DC-offset used in this file. If None,
                       the value is taken from self.kat.data['DCoffset']
                       DCoffset = {'DARM': value [deg], 'MICH': value [deg]}
        tol          - Relative tolerance for the RoCs and focal lengths.
        lesning      - If true, input mirror thermal lensing effect is included
        RoC          - If true, test mass RoC changes are included
        verbose      - If set to True, information is printed.
        """
        
        kat1 = self.kat.deepcopy()
        if DCoffset is None:
            DCoffset = self.kat.data['DCoffset']
        elif not isinstance(DCoffset, dict):
            raise pkex.BasePyKatException(("DCoffset is of type {}. Must be None or dictionary, "+
                                           "e.g., {{'DARM': value [deg]}}").format(type(DCoffset)))

        # Cold IFO RoCs and focal lengthts
        new = copy.copy(kat1.data['cold_optics_parameters'])
        old1 = copy.copy(new)
        run = True
        a = 0
        # Iteratively finding the warm IFO parameters
        while run:
            a += 1
            vprint(verbose, 'Iteration {}'.format(a))
            # Keeps track of parameters from the two previous runs
            old2 = copy.copy(old1)
            old1 = copy.copy(new)
            # Finding the needed maxtem
            vprint(verbose, ' Finding maxtem...',end=" ")
            kat2 = kat1.deepcopy()
            kat2.IFO.remove_modulators()
            kat2.IFO.find_maxtem(tol=1e-4)
            kat1.maxtem = kat2.maxtem
            # Re-tuning IFO
            vprint(verbose, '{}\n Re-tuning interferometer...'.format(kat2.maxtem), end=" ")
            pretune(kat2, 1e-7, verbose=False)
            kat1.IFO.apply_tunings(kat2.IFO.get_tunings())
            if 'DCoffset' in kat1.data:
                kat1.data['DCoffset'] = {}
            # Setting DC-offset
            for k,v in DCoffset.items():
                kat1.IFO.set_DC_offset(DCoffset=v, offset_type=k, verbose=False)
            vprint(verbose, 'Done!\n Computing thermal effect...', end=" ")
            # Computing the thermal effect
            new, out = compute_thermal_effect(kat1, mirror_list, lensing = lensing, RoC = RoC)
            vprint(verbose, 'Done!')

            # Relative differences between new and previous parameters
            diff1 = np.zeros(len(new), dtype=float)
            diff2 = np.zeros(len(new), dtype=float)
            for i,(k,v) in enumerate(new.items()):
                diff1[i] = np.abs((v - old1[k])/old1[k])
                diff2[i] = np.abs((v - old2[k])/old2[k])

            if diff1.max() < tol:
                # Stop running if diff1 is small
                run = False
            elif diff2.max() < tol*1e-1:
                # Taking a half step if diff2 is small
                for i,(k,v) in enumerate(new.items()):
                    new[k] = (v+old1[k])/2.0
                    print("{:.10f}, {:.10f},  {:.10f},  {:.10f}".format(old2[k], old1[k], v, new[k]))

            # Updating IFO parameters
            for i,(k,v) in enumerate(new.items()):
                # Updating IFO parameters
                if isinstance(kat1.components[k], pykat.components.lens):
                    kat1.components[k].f = v 
                    vprint(verbose, '  {}.f: {:>11.5e} m --> {:>8.5e} m'.format(k, kat2.components[k].f.value,
                                                                          kat1.components[k].f.value))
                elif (isinstance(kat1.components[k], pykat.components.mirror) or 
                      isinstance(kat1.components[k], pykat.components.beamSplitter)):
                    kat1.components[k].Rc = v
                    vprint(verbose, '  {}.Rc: {:>14.3f} m --> {:>11.3f} m'.format(k, kat2.components[k].Rc.value,
                                                                           kat1.components[k].Rc.value))
            vprint(verbose and not run, ' Converged!')            

        # Setting new parameters to the kat-object
        for i,(k,v) in enumerate(new.items()):
            if isinstance(kat1.components[k], pykat.components.lens):
                self.kat.components[k].f = v 
            elif (isinstance(kat1.components[k], pykat.components.mirror) or 
                  isinstance(kat1.components[k], pykat.components.beamSplitter)):
                self.kat.components[k].Rc = v
                
        self.kat.maxtem = kat1.maxtem
        kat = self.kat.deepcopy()
        kat.IFO.remove_modulators()
        pretune(kat, 1e-7, verbose=False)
        self.apply_tunings(kat.IFO.get_tunings())
        if 'DCoffset' in self.kat.data:
                self.kat.data['DCoffset'] = {}
        for k,v in DCoffset.items():                
            self.set_DC_offset(DCoffset=v, offset_type=k, verbose=False)

    def _strToDOFs(self, DOFs):
        dofs = []

        for _ in DOFs:
            if isinstance(_, six.string_types):
                if _ in self.DOFs:
                    dofs.append(self.DOFs[_])
                else:
                    raise pkex.BasePyKatException(
                        "Could not find DOF called `%s`. Possible DOF options: %s" % (
                        _, str(list(self.DOFs.keys()))))
            else:
                raise pkex.BasePyKatException(
                    "'%s' not possible DOF options: %s" % (
                    _, str(list(self.DOFs.keys()))))

        return dofs

def assert_advplus_ifo_kat(kat):
    # TODO update to advplus if needed
    if not isinstance(kat.IFO, ADVPLUS_IFO):
        raise pkex.BasePyKatException("\033[91mkat file is not an ADVPLUS_IFO compatiable kat\033[0m")
              


def make_kat(name="avirgo_PR_OMC", katfile=None, verbose=False, debug=False, keepComments=False, preserveConstants=True):
    """
    Returns a kat object and fills in the kat.IFO property for storing
    the associated interferometer data.
    
    The `name` argument selects from default Advanced Virgo files included in Pykat:
    TODO add names for with/without OMC and for O4/O5 scenarios    

    keepComments: If true it will keep the original comments from the file
    preserveComments: If true it will keep the const commands in the kat
    """
    # TODO review and describe file options, provide default 'design' option.
    # Pre-defined file-names
    names = ['avirgoplus_design1']
    
    # TODO this is not as usual as initially though. Should be removed.
    # Mirror names. Mapping to IFO-specific names to faciliate creating new IFO-specific files.
    # Change the values in the dictionary to the IFO-specific mirror names. Do not change the
    # keys, they are used in functions and methods.
    """
    mirrors = {'EX': 'NE', 'EY': 'WE',
               'EXAR': 'NEAR', 'EYAR': 'WEAR',
               'IX': 'NI', 'IY': 'WI',
               'IXAR': 'NIAR', 'IYAR': 'WIAR',
               'PRM': 'PR', 'SRM': 'SR',
               'PRMAR': 'PRAR', 'SRMAR': 'SRAR',
               'PR2': None, 'PR3': None,
               'SR2': None, 'SR3': None,
               'BS': 'BS', 'BSARX': 'BSAR1', 'BSARY': 'BSAR2'}
    """

    # TODO overwirte to flush out use of mapped mirrors
    mirrors = ['NE', 'NI', 'NIAR', 'WE', 'WI', 'WIAR', 'PR', 'PRAR', 'SR', 'SRAR', 'BS', 'BSAR1', 'BSAR2']

    # Other useful components to map in the same way as the mirrors

    #signalNames = {'AS_DC': 'B1_DC', 'POP_f1': 'B2_f1', 'POP_f2': 'B2_f2', 'POP_f3': 'B2_f3', 'POP_f4':
    #               'B2_f4', 'REFL_f1': 'B4_f1', 'REFL_f2': 'B4_f2'}

    # nodes = {}

    # TODO are these used?
    etalonW = 0
    etalonN = 0
    # Define which mirrors create the tuning description. Has to be consistent
    # with values in the mirrors dictionary above. 
    tunings_components_list = ["PR", "NI", "NE", "WI", "WE", "BS", "SR"]

    # Define which keys are used for a tuning description
    tuning_keys_list = ["maxtem", "phase"]

    if debug:
        kat = finesse.kat(tempdir=".",tempname="test")
    else:
        kat = finesse.kat()
    
    kat.verbose=verbose
    
    files_directory = pkg_resources.resource_filename('pykat.ifo', os.path.join('adv','files'))
    
    if katfile:
        kat.load(katfile, keepComments=keepComments, preserveConstants=preserveConstants)
    else:
        if name not in names:
            pkex.printWarning("adv name `{}' not recognised, options are {}, using default 'design'".format(name, names))
        
        katfile = os.path.join(files_directory, name+".kat")
        kat.load(katfile, keepComments=keepComments, preserveConstants=preserveConstants)


    # Removing SR if it isn't in the kat-file, or if it's fully transparent.
    isSRC = True
    if not 'SR' in kat.components:
        isSRC = False
        tunings_components_list.pop(tunings_components_list.index('SR'))
    elif kat.components['SR'].R.value == 0:
        isSRC = False
        tunings_components_list.pop(tunings_components_list.index('SR'))
            
    # Checking if mirrors in tuning_component_list are in the kat-object
    for m in tunings_components_list:
        if m in kat.components:
            if not ( isinstance(kat.components[m], pykat.components.mirror) or
                     isinstance(kat.components[m], pykat.components.beamSplitter) ):
                raise pkex.BasePyKatException('{} is not a mirror or beam splitter'.format(m))
        else:
            raise pkex.BasePyKatException('{} is not a component in the kat-object'.format(m))

    # Checking if mirrors in mirrors-dictionary are in the kat-object.
    for comp in mirrors:
        if comp in kat.components:
            if not ( isinstance(kat.components[comp], pykat.components.mirror) or
                     isinstance(kat.components[comp], pykat.components.beamSplitter) ):
                raise pkex.BasePyKatException('{} is not a mirror or a beam splitter'.format(v))
        elif not comp is None:
            # Allowing SR mirrors to be in the mirrors-dictionary anyway if isSRC = False
            if not ( (comp == 'SR' and not isSRC) or (comp == 'SRAR' and not isSRC) ):
                raise pkex.BasePyKatException('{} is not a component in the kat-object'.format(v))


    
    # Creating the IFO object
    kat.IFO = ADVPLUS_IFO(kat, tuning_keys_list, tunings_components_list)
    kat.IFO._data_path = files_directory
    kat.IFO.rawBlocks = BlockedKatFile()
    kat.IFO.rawBlocks.read(katfile)
    kat.IFO.isSRC = isSRC
    kat.IFO.mirrors = mirrors

    ## # -------
    
    ## # Create empty object to just store whatever DOFs, port, variables in
    ## # that will be used by processing functions
    ## kat.IFO = ADV_IFO(kat, tuning_keys_list, tunings_components_list)

    
    ## kat.IFO._data_path=pkg_resources.resource_filename('pykat.ifo', os.path.join('adv','files'))

    ## kat.IFO.rawBlocks = BlockedKatFile()
    
    ## if katfile:
    ##     kat.load(katfile, keepComments=keepComments, preserveConstants=preserveConstants)
    ##     kat.IFO.rawBlocks.read(katfile)
    ## else:
    ##     if name not in names:
    ##         pkex.printWarning("adv name `{}' not recognised, options are {}, using default 'design'".format(name, names))
        
    ##     katkile = os.path.join(kat.IFO._data_path, name+".kat")
        
    ##     kat.load(katkile, keepComments=keepComments, preserveConstants=preserveConstants)
    ##     kat.IFO.rawBlocks.read(katkile)

    ## # --------

    # ----------------------------------------------------------------------
    # get and derive parameters from the kat file

    
    #f1 = 6270777            # fmod1 in TDR
    #f3 = 8361036            # 4 / 3 * f1, fmod3 in TDR
    #f2 = 56436993           # 9 * f1, fmod2 in TDR
    #f4 = 119144763.0        # 19 * f1, new f4.
    #f4b = 131686317         # 21 * f1, fmod4 in TDR. Old f4.
    
    # Get main sideband frequencies
    if "f1" in kat.constants.keys():
        kat.IFO.f1 = float(kat.constants["f1"].value)
    else:
        kat.IFO.f1 = 6270777.0
        
    if "f2" in kat.constants.keys():
        kat.IFO.f2 = float(kat.constants["f2"].value)
    else:
        kat.IFO.f2 = 56436993.0
        
    if "f3" in kat.constants.keys():
        kat.IFO.f3 = float(kat.constants["f3"].value)
    else:
        kat.IFO.f3 = 8361036.0

    # if "f4" in kat.constants.keys():
    #     kat.IFO.f4 = float(kat.constants["f4"].value)
    # else:
    #     kat.IFO.f4 = 119144763.0

    # if "f4b" in kat.constants.keys():
    #     kat.IFO.f4b = float(kat.constants["f4b"].value)
    # else:
    #     kat.IFO.f4b = 131686317.0
    
    # kat.IFO.f36M = kat.IFO.f2 - kat.IFO.f1
        
    # TODO add else here!
    # check modultion frequencies
    #if (5 * kat.IFO.f1 != kat.IFO.f2):
    #    print(" ** Warning: modulation frequencies do not match: 5*f1!=f2")
    
    # defining a dicotionary for the main mirror positions (tunings),
    # keys should include maxtem, phase and all main optics names
    #kat.IFO.tunings = get_tunings(dict.fromkeys(["maxtem", "phase", "PR", "NI", "NE", "WI", "WE", "BS", "SRM"]))
    kat.IFO.compute_derived_lengths()
        
    # ----------------------------------------------------------------------
    # define ports and signals 
    
    # Useful signals
    kat.IFO.B1   = Output(kat.IFO, "B1", "nB1")
    kat.IFO.B1p = Output(kat.IFO, "B1p", "nSR2")
    kat.IFO.B1p_f2 = Output(kat.IFO, "B1p_f2", "nSR2", "f2", phase = 31.1)

    kat.IFO.B2_f1 = Output(kat.IFO, "B2_f1", "nB2", "f1", phase = 170.1)
    kat.IFO.B2_f2 = Output(kat.IFO, "B2_f2", "nB2", "f2", phase = -73.8)
    kat.IFO.B2_f3 = Output(kat.IFO, "B2_f3", "nB2", "f3", phase = -4.5)
    # kat.IFO.B2_f4 = Output(kat.IFO, "B2_f4", "nB2", "f4", phase = 0)
    # kat.IFO.B2_f4b = Output(kat.IFO, "B2_f4b", "nB2", "f4b", phase = 0)

    
    kat.IFO.B4_f1  = Output(kat.IFO, "B4_f1",  "nB4",  "f1", phase = 173)
    kat.IFO.B4_f2  = Output(kat.IFO, "B4_f2",  "nB4",  "f2", phase = 32.6)
    
    kat.IFO.POW_BS  = Output(kat.IFO, "PowBS", "nBSs*")
    kat.IFO.POW_X   = Output(kat.IFO, "PowN",  "nNI2")
    kat.IFO.POW_Y   = Output(kat.IFO, "PowW",  "nWI2")
    if isSRC:
        kat.IFO.POW_S   = Output(kat.IFO, "PowS",  "nSR1")

    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # TODO add AR syrfaces to DOFs to avoid etalon mistuning 

    # Pretune LSC DOF
    kat.IFO.preARMN =  DOF(kat.IFO, "preARMN", kat.IFO.POW_X,   "", ["NE"], 1, 1.0, sigtype="z")
    kat.IFO.preARMW =  DOF(kat.IFO, "preARMW", kat.IFO.POW_Y,   "", ["WE"], 1, 1.0, sigtype="z")
    kat.IFO.preMICH =  DOF(kat.IFO, "preMICH"  , kat.IFO.B1,   "", ["NI","NE","WI","WE"], [-1,-1,1,1], 6.0, sigtype="z")
    kat.IFO.prePRCL =  DOF(kat.IFO, "prePRCL", kat.IFO.POW_BS,  "", ["PR"],  1, 50, sigtype="z")
    kat.IFO.preDARM = DOF(kat.IFO, "preDARM", kat.IFO.POW_X, "", ["NE","WE"], [-1,1], 1.0, sigtype="z")
    kat.IFO.preCARM = DOF(kat.IFO, "preCARM", kat.IFO.POW_X, "", ["NE","WE"], [-1,-1], 1.0, sigtype="z")
    if isSRC:
        kat.IFO.preSRCL =  DOF(kat.IFO, "preSRCL", kat.IFO.B1,   "", "SR",  1, 50.0, sigtype="z")
    

    # New signals following VIR-0445A-20
    # only DF mod with B1p for the moment
    # TODO crosscheck and add other options
    kat.IFO.PRCL =  DOF(kat.IFO, "PRCL", kat.IFO.B2_f3, "I", ["PR"], 1, 100.0, sigtype="z")
    kat.IFO.MICH =  DOF(kat.IFO, "MICH", kat.IFO.B2_f2, "Q", ["NI","NE","WI","WE"],[-1,-1,1,1], 100.0, sigtype="z")
    kat.IFO.CARM =  DOF(kat.IFO, "CARM", kat.IFO.B2_f1, "I", ["NE","WE"], [-1, -1], 1.5, sigtype="z")
    #kat.IFO.DARM =  DOF(kat.IFO, "DARM", kat.IFO.B1p_f2, "I", ["NE","WE"], [-1,1], 1.0, sigtype="z")
    #kat.IFO.DARM_h =  DOF(kat.IFO, "DARM_h", kat.IFO.B1p_f2, "I", ["LN","LW"], [-1,1], 1.0, sigtype="z")
    kat.IFO.DARM =  DOF(kat.IFO, "DARM", kat.IFO.B1,   "", ["NE","WE"], [-1,1], 1.0, sigtype="z")
    kat.IFO.DARM_h =  DOF(kat.IFO, "DARM_h", kat.IFO.B1,   "",  ["LN", "LW"], [-1,1], 1.0, sigtype="phase")
    if isSRC:
        kat.IFO.SRCL =  DOF(kat.IFO, "SRCL", kat.IFO.B2_f2, "I", ["SR"], -1, 1e2, sigtype="z")

    kat.IFO.LSC_DOFs = (kat.IFO.PRCL, kat.IFO.MICH, kat.IFO.CARM, kat.IFO.DARM)
    if isSRC:
        kat.IFO.LSC_DOFs = kat.IFO.LSC_DOFs + (kat.IFO.SRCL,)
        
    kat.IFO.CAV_POWs = (kat.IFO.POW_X, kat.IFO.POW_Y, kat.IFO.POW_BS)
    if isSRC:
        kat.IFO.CAV_POWs = kat.IFO.CAV_POWs + (kat.IFO.POW_S,)

    #################################################################
    # Pitch DOfs (not yet Virgo compatible. Below code is for Ligo.)
    #################################################################
    # There is a difference in the way LIGO and Finesse define positive and negative
    # rotations of the cavity mirrors. For LIGO the rotational DOFs assume ITM + rotation
    # is clockwise and ETM + rotation is anticlockwise.
    # I'll be explict here for future reference.
    # TODO check why AR surfaces are required here and below, maybe can use put like for etalons?
    cav_mirrors = ["NE", "NEAR", "WE", "WEAR", "NI", "NIAR", "WI", "WIAR"]

    # LIGO definitions
    # Based on figure 7 in T0900511-v4
    CHARD_factors   = np.array([ 1, 1, 1, 1,-1,-1,-1,-1])
    DHARD_factors   = np.array([ 1, 1,-1,-1,-1,-1, 1, 1])
    CSOFT_factors   = np.array([-1,-1,-1,-1,-1,-1,-1,-1])
    # DSOFT_factors   = np.array([-1,-1, 1, 1, 1, 1,-1,-1])   # Wrong!
    DSOFT_factors   = np.array([-1,-1, 1, 1,-1,-1, 1, 1])
    
    # Finesse definitions
    # negative for ITM rotations
    ITMS = np.in1d(cav_mirrors, np.array(["NI", "NIAR", "WI", "WIAR"]))
    CHARD_factors[ITMS] *= -1
    DHARD_factors[ITMS] *= -1
    CSOFT_factors[ITMS] *= -1
    DSOFT_factors[ITMS] *= -1

    kat.IFO.CHARD_P = DOF(kat.IFO, "CHARD_P", None , None, cav_mirrors, CHARD_factors, 1, sigtype="pitch")
    kat.IFO.DHARD_P = DOF(kat.IFO, "DHARD_P", None , None, cav_mirrors, DHARD_factors, 1, sigtype="pitch")
    kat.IFO.CSOFT_P = DOF(kat.IFO, "CSOFT_P", None , None, cav_mirrors, CSOFT_factors, 1, sigtype="pitch")
    kat.IFO.DSOFT_P = DOF(kat.IFO, "DSOFT_P", None , None, cav_mirrors, DSOFT_factors, 1, sigtype="pitch")
    kat.IFO.PR_P   = DOF(kat.IFO, "PR_P"  , None , None, ["PR", "PRAR"], [1,1], 1, sigtype="pitch")
    
    # TDOO clean this up
    """
    if not mirrors["PR2"] is None:
        kat.IFO.PR2_P  = DOF(kat.IFO, "PRC2_P" , None , None, mirrors["PR2"], [1], 1, sigtype="pitch")
    if not mirrors["PR3"] is None:
        kat.IFO.PR3_P  = DOF(kat.IFO, "PRC3_P" , None , None, mirrors["PR3"], [1], 1, sigtype="pitch")
    """
    if isSRC:
        kat.IFO.SR_P = DOF(kat.IFO, "SR_P"  , None , None, ["SR", "SRAR"], [1,1], 1, sigtype="pitch")
    """
        if not mirrors['SR2'] is None:
            kat.IFO.SR2_P  = DOF(kat.IFO, "SR2_P" , None , None, mirrors["SR2"], [1], 1, sigtype="pitch")
        if not mirrors['SR3'] is None:
            kat.IFO.SR3_P  = DOF(kat.IFO, "SR3_P" , None , None, mirrors["SR3"], [1], 1, sigtype="pitch")
    """

    kat.IFO.MICH_P  = DOF(kat.IFO, "MICH_P" , None , None, ["BS", "BSAR1", "BSAR2"],
                          [1,1,1], 1, sigtype="pitch")
    
    kat.IFO.ASC_P_DOFs = (kat.IFO.CHARD_P, kat.IFO.DHARD_P,
                          kat.IFO.CSOFT_P, kat.IFO.DSOFT_P,
                          kat.IFO.PR_P, kat.IFO.MICH_P)
    
    # Adding SRC pitch DoFs if SRC is included
    if isSRC:
        kat.IFO.ASC_P_DOFs = kat.IFO.ASC_P_DOFs + (kat.IFO.SR_P,)
        """
        if not mirrors["SR2"] is None:
            kat.IFO.ASC_P_DOFs = kat.IFO.ASC_P_DOFs + (kat.IFO.SR2_P,)
        if not mirrors["SR3"] is None:
            kat.IFO.ASC_P_DOFs = kat.IFO.ASC_P_DOFs + (kat.IFO.SR3_P,)
        """
    """    
    # Adding PR2 and PR3 pitch if they are included
    if not mirrors["PR2"] is None:
        kat.IFO.ASC_P_DOFs = kat.IFO.ASC_P_DOFs + (kat.IFO.PR2_P,)
    if not mirrors["PR3"] is None:
        kat.IFO.ASC_P_DOFs = kat.IFO.ASC_P_DOFs + (kat.IFO.PR3_P,)
    """
        
    ##########################
    # For thermal effects
    ##########################
        
    if not 'mirror_properties' in kat.data:
        # Mirror properties for computing thermal effects
        MPs = {}
        # Common properties. Check if this is true.
        common_properties = {}
        common_properties['K'] = 1.380        # Thermal conductivity. Check value!
        common_properties['T0'] = 295.0       # Temperature. Check value!
        common_properties['emiss'] = 0.89     # Emissivity. Check value!
        common_properties['alpha'] = 0.54e-6  # Thermal expansion coeff. Check value!
        common_properties['sigma'] = 0.164    # Poisson ratio. Check value!
        common_properties['dndT'] = 8.7e-6    # dn/dT. Check value!
        # TODO check if this works with new 'mirrors' list
        # Setting common propertis
        for m in mirrors:
            if not ('AR' in m):
                # print(k)
                MPs[m] = copy.deepcopy(common_properties)

        # Setting mirror specific properties
        # HR coating absorptions. Values from Valeria Sequino. 
        MPs['WI']['aSub'] = 3.0e-5
        MPs['NE']['aCoat'] = 0.24e-6
        MPs['WE']['aCoat'] = 0.24e-6
        MPs['NI']['aCoat'] = 0.19e-6
        MPs['WI']['aCoat'] = 0.28e-6
        # Substrate absorption [1/m]. Using upper limits from [TDR, table 2.6].
        MPs['NE']['aSub'] = 3.0e-5
        MPs['WE']['aSub'] = 3.0e-5
        MPs['NI']['aSub'] = 3.0e-5
        MPs['NI']['aSub'] = 3.0e-5
        MPs['NI']['aSub'] = 3.0e-5

        kat.IFO.mirror_properties = MPs

    
    # Storing RoCs and focal lengths for the cold IFO. To be used when computing thermal effects
    if not 'cold_optics_parameters' in kat.data:
        cold = {}
        for k,v in kat.components.items():
            if isinstance(v, pykat.components.mirror) or isinstance(v, pykat.components.beamSplitter):
                cold[k] = v.Rc.value
            elif isinstance(v, pykat.components.lens):
                cold[k] = v.f.value

        # print(kat.IFO.cold_ifo)
        # kat.IFO.cold_ifo = cold
        kat.data['cold_optics_parameters'] = cold

    kat.IFO.update()
    kat.IFO.lockNames = None
    
    return kat
    
#def scan_to_precision(kat, DOF, pretune_precision, minmax="max", phi=0.0, precision=60.0):
#    assert_advplus_ifo_kat(kat)
#    
#    while precision > pretune_precision * DOF.scale:
#        out = scan_DOF(kat, DOF, xlimits = [phi-1.5*precision, phi+1.5*precision])
#        phi, precision = find_peak(out, DOF.port.name, minmax=minmax)
#        
#    return phi, precision
        
def pretune(_kat, pretune_precision=1.0e-4, verbose=False):
    assert_advplus_ifo_kat(_kat)

    # This function needs to apply a bunch of pretunings to the original
    # kat and associated IFO object passed in

    vprint(verbose,"-- pretuning interferometer to precision {0:2g} deg = {1:2g} m".format(pretune_precision,
                                                                                           pretune_precision*_kat.lambda0/360.0))
    kat = _kat.deepcopy()
    kat.removeBlock("locks", False)
    null_modulation_index(kat)

    vprint(verbose, "   scanning X arm (maximising power)")
    
    make_transparent(kat, ["PR"])
    if kat.IFO.isSRC:
        make_transparent(kat,["SR"])
    make_transparent(kat, ["WI","WIAR","WE"])
    
    kat.BS.setRTL(0.0, 1.0, 0.0) # set BS refl. for X arm
    
    phi, precision = scan_to_precision(kat.IFO.preARMN, pretune_precision)
    phi = round(phi/pretune_precision)*pretune_precision
    phi = round_to_n(phi,5)
    
    vprint(verbose, "   found max/min at: {} (precision = {:2g})".format(phi, precision))
    
    _kat.IFO.preARMN.apply_tuning(phi)

    vprint(verbose, "   scanning Y arm (maximising power)")
    kat = _kat.deepcopy()
    kat.removeBlock("locks", False)
    null_modulation_index(kat)
    
    make_transparent(kat,["PR"])
    if kat.IFO.isSRC:
        make_transparent(kat,["SR"])
    make_transparent(kat,["NI","NIAR","NE"])
    kat.BS.setRTL(1.0,0.0,0.0) # set BS refl. for Y arm
    phi, precision = scan_to_precision(kat.IFO.preARMW, pretune_precision)
    phi=round(phi/pretune_precision)*pretune_precision
    phi=round_to_n(phi,5)
    vprint(verbose, "   found max/min at: {} (precision = {:2g})".format(phi, precision))
    _kat.IFO.preARMW.apply_tuning(phi)

    vprint(verbose, "   scanning MICH (minimising power)")
    kat = _kat.deepcopy()
    kat.removeBlock("locks", False)
    null_modulation_index(kat)
    
    make_transparent(kat,["PR"])
    if kat.IFO.isSRC:
        make_transparent(kat,["SR"])
    phi, precision = scan_to_precision(kat.IFO.preMICH, pretune_precision, minmax="min", precision=30, relative=True)
    phi=round(phi/pretune_precision)*pretune_precision
    phi=round_to_n(phi,5)
    vprint(verbose, "   found max/min at: {} (precision = {:2g})".format(phi, precision))
    _kat.IFO.preMICH.apply_tuning(phi, add=True)

    vprint(verbose, "   scanning PRCL (maximising power)")
    kat = _kat.deepcopy()
    kat.removeBlock("locks", False)
    null_modulation_index(kat)
    if kat.IFO.isSRC:
        make_transparent(kat,["SR"])
    #print(kat)
    phi, precision = scan_to_precision(kat.IFO.prePRCL, pretune_precision)
    phi=round(phi/pretune_precision)*pretune_precision
    phi=round_to_n(phi,5)
    vprint(verbose, "   found max/min at: {} (precision = {:2g})".format(phi, precision))
    _kat.IFO.prePRCL.apply_tuning(phi)

    if _kat.IFO.isSRC:
        vprint(verbose, "   scanning SRCL (maximising carrier power, then adding 90 deg)")
        kat = _kat.deepcopy()
        kat.removeBlock("locks", False)
        null_modulation_index(kat)

        phi, precision = scan_to_precision(kat.IFO.preSRCL, pretune_precision, phi=0, precision = 10)
        phi=round(phi/pretune_precision)*pretune_precision
        phi=round_to_n(phi,4)-90.0

        vprint(verbose, "   found max/min at: {} (precision = {:2g})".format(phi, precision))
        _kat.IFO.preSRCL.apply_tuning(phi)

    # Removing previously set DCoffset dictionary
    _kat.data['DCoffset'] = {}
    
    vprint(verbose,"   ... done")

def pretune_status(_kat):
    assert_advplus_ifo_kat(_kat)
    
    kat = _kat.deepcopy()
    kat.verbose = False
    kat.noxaxis = True
    
    pretune_DOFs = [kat.IFO.preARMN, kat.IFO.preARMW, kat.IFO.prePRCL, kat.IFO.preMICH]
    if kat.IFO.isSRC:
        pretune_DOFs.append(kat.IFO.preSRCL)
    
    _detStr=""
    
    for dof in pretune_DOFs:
        dof.add_signal()
        
    out = kat.run()
    Pin = float(kat.i1.P)

    tunings = kat.IFO.get_tunings()
    
    if tunings['keys']["maxtem"] == -1:
        _maxtemStr="off"
    else:
        _maxtemStr = "{:3}".format(tunings['keys']["maxtem"])
        
    print(" .---------------------------------------------------.")
    print(" | pretuned for maxtem = {}, phase = {:2}             |".format(_maxtemStr, int(kat.phase)))
    
    keys_t = list(tunings.keys())
    keys_t.remove("keys")
    
    print(" .---------------------------------------------------.")
    print(" | port    power[W] pow. ratio | optics   tunings    |")
    print(" +-----------------------------|---------------------+")
    
    idx_p = 0
    idx_t = 0
    
    while (idx_p < len(pretune_DOFs) or idx_t < len(keys_t)):
        if idx_p < len(pretune_DOFs):
            p = pretune_DOFs[idx_p]
            print(" | {:6}: {:9.4g} {:9.4g} |".format(p.port.name, float(out[p.port.name]), float(out[p.port.name])/Pin),end="")
            idx_p +=1
        else:
            print(" |                             |", end="")
            
        if idx_t < len(keys_t):
            t=keys_t[idx_t]
            print(" {:6}: {:9.4g}   |".format(t, float(tunings[t])))
            idx_t +=1
        else:
            print("                     |")
            
    print(" `---------------------------------------------------'")

# probably extra and can be removed
def power_ratios(_kat):
    assert_advplus_ifo_kat(_kat)
    
    kat = _kat.deepcopy()
    kat.verbose = False
    kat.noxaxis = True

    ports = [kat.IFO.POW_X, kat.IFO.POW_Y, kat.IFO.B1, kat.IFO.POW_BS]
    _detStr = ""
    
    for p in ports:
        _sigStr = p.signal(kat)
        _detStr = "\n".join([_detStr, _sigStr])
    
    kat.parse(_detStr)
    
    out = kat.run()
    
    Pin = float(kat.i1.P)

    print("-- power ratios (Pin = {0:.3g} W)".format(Pin))
    
    for p in ports:
        print(" {0:6} = {1:8.3g} W ({0:6}/Pin = {2:8.2g})" .format(p.name, float(out[p.name]), float(out[p.name])/Pin))


def generate_locks(kat, gainsAdjustment = [0.1, 0.9, 0.9, 0.001, 0.02],
                    gains=None, accuracies=None,
                    rms=[1e-14, 1e-14, 1e-12, 1e-11, 50e-11], verbose=True,
                    useDiff = True):
    """
    gainsAdjustment: factors to apply to loop gains computed from optical gains
    gains:           override loop gain [W per deg]
    accuracies:      overwrite error signal threshold [W]
    useDiff:         use diff command instead of fsig to compute optical gains
                    
    rms: loop accuracies in meters (manually tuned for the loops to work
         with the default file)
         to compute accuracies from rms, we convert
         rms to radians as rms_rad = rms * 2 pi/lambda
         and then multiply by the optical gain.
                    
    NOTE: gainsAdjustment, gains, accuracies and rms are specified in the order of DARM, CARM, PRCL, MICH, SRCL.
    """
    assert_advplus_ifo_kat(kat)
        
    # optical gains in W/rad
    
    ogDARM = optical_gain(kat.IFO.DARM, kat.IFO.DARM, useDiff=useDiff)
    ogCARM = optical_gain(kat.IFO.CARM, kat.IFO.CARM, useDiff=useDiff)
    ogPRCL = optical_gain(kat.IFO.PRCL, kat.IFO.PRCL, useDiff=useDiff)
    ogMICH = optical_gain(kat.IFO.MICH, kat.IFO.MICH, useDiff=useDiff)
    if kat.IFO.isSRC:
        ogSRCL = optical_gain(kat.IFO.SRCL, kat.IFO.SRCL, useDiff=useDiff)

    if gains is None:            
        # manually tuning relative gains
        factor = -1.0 * 180 / math.pi # convert from rad/W to -1 * deg/W
        
        gainDARM = round_to_n(gainsAdjustment[0] * factor / ogDARM, 2) # manually tuned
        gainCARM = round_to_n(gainsAdjustment[1] * factor / ogCARM, 2) # factor 0.005 for better gain hirarchy with DARM
        gainPRCL = round_to_n(gainsAdjustment[2] * factor / ogPRCL, 2) # manually tuned
        gainMICH = round_to_n(gainsAdjustment[3] * factor / ogMICH, 2) # manually tuned
        gains = [ gainDARM, gainCARM, gainPRCL, gainMICH]
        if kat.IFO.isSRC:
            gainSRCL = round_to_n(gainsAdjustment[4] * factor / ogSRCL, 2) # gain hirarchy with MICH
            gains.append(gainSRCL)
    
    if accuracies is None:
        factor = 2.0 * math.pi / kat.lambda0 # convert from m to radians
        
        accDARM = round_to_n(np.abs(factor * rms[0] * ogDARM), 2) 
        accCARM = round_to_n(np.abs(factor * rms[1] * ogCARM), 2)
        accPRCL = round_to_n(np.abs(factor * rms[2] * ogPRCL), 2)
        accMICH = round_to_n(np.abs(factor * rms[3] * ogMICH), 2)
        accuracies = [accDARM, accCARM, accPRCL, accMICH]
        if kat.IFO.isSRC:
            accSRCL = round_to_n(np.abs(factor * rms[4] * ogSRCL), 2)
            accuracies.append(accSRCL)
            
    factor1 = 2.0 * math.pi / 360.0 
    factor2 = 2.0 * math.pi / kat.lambda0 
    factor3 = 360.0  / kat.lambda0
    factor4 = -1.0 * 180 / math.pi 

    if verbose:
        print(" .--------------------------------------------------.")
        print(" | Parameters for locks:                            |")
        print(" +--------------------------------------------------+")
        print(" | -- optical gains [W/rad], [W/deg] and [W/m]:     |")
        print(" | DARM: {:12.5}, {:12.5}, {:12.5}   |".format(ogDARM, ogDARM*factor1, ogDARM*factor2))
        print(" | CARM: {:12.5}, {:12.5}, {:12.5}   |".format(ogCARM, ogCARM*factor1, ogCARM*factor2))
        print(" | PRCL: {:12.5}, {:12.5}, {:12.5}   |".format(ogPRCL, ogPRCL*factor1, ogPRCL*factor2))
        print(" | MICH: {:12.5}, {:12.5}, {:12.5}   |".format(ogMICH, ogMICH*factor1, ogMICH*factor2))
        if kat.IFO.isSRC:
            print(" | SRCL: {:12.5}, {:12.5}, {:12.5}   |".format(ogSRCL, ogSRCL*factor1, ogSRCL*factor2))
        print(" +--------------------------------------------------+")
        print(" | -- defult loop accuracies [deg], [m] and [W]:    |")
        print(" | DARM: {:12.6}, {:12.6}, {:12.6}   |".format(factor3*rms[0], rms[0], np.abs(rms[0]*ogDARM*factor2)))
        print(" | CARM: {:12.6}, {:12.6}, {:12.6}   |".format(factor3*rms[1], rms[1], np.abs(rms[1]*ogCARM*factor2)))
        print(" | PRCL: {:12.6}, {:12.6}, {:12.6}   |".format(factor3*rms[2], rms[2], np.abs(rms[2]*ogPRCL*factor2)))
        print(" | MICH: {:12.6}, {:12.6}, {:12.6}   |".format(factor3*rms[3], rms[3], np.abs(rms[3]*ogMICH*factor2)))
        if kat.IFO.isSRC:
            print(" | SRCL: {:12.6}, {:12.6}, {:12.6}   |".format(factor3*rms[4], rms[4], np.abs(rms[4]*ogSRCL*factor2)))
        print(" +--------------------------------------------------+")
        print(" | -- extra gain factors (factor * 1/optical_gain): |")
        print(" | DARM: {:5.4} * {:12.6} = {:12.6}        |".format(gainsAdjustment[0],factor4/ogDARM, gainsAdjustment[0]*factor4/ogDARM))
        print(" | CARM: {:5.4} * {:12.6} = {:12.6}        |".format(gainsAdjustment[1],factor4/ogCARM, gainsAdjustment[1]*factor4/ogCARM))
        print(" | PRCL: {:5.4} * {:12.6} = {:12.6}        |".format(gainsAdjustment[2],factor4/ogPRCL, gainsAdjustment[2]*factor4/ogPRCL))
        print(" | MICH: {:5.4} * {:12.6} = {:12.6}        |".format(gainsAdjustment[3],factor4/ogMICH, gainsAdjustment[3]*factor4/ogMICH))
        if kat.IFO.isSRC:
            print(" | SRCL: {:5.4} * {:12.6} = {:12.6}        |".format(gainsAdjustment[4],factor4/ogSRCL, gainsAdjustment[4]*factor4/ogSRCL))
        print(" `--------------------------------------------------'")
        
    data = {
        "DARM": {"accuracy": accuracies[0], "gain": gains[0]},
        "CARM": {"accuracy": accuracies[1], "gain": gains[1]},
        "PRCL": {"accuracy": accuracies[2], "gain": gains[2]},
        "MICH": {"accuracy": accuracies[3], "gain": gains[3]}
        }
    if kat.IFO.isSRC:
        data['SRCL'] = {"accuracy": accuracies[4], "gain": gains[4]}
    
    return data


def compute_thermal_effect(kat, mirror_list, lensing = True, RoC = True):
    '''
    Computes the thermal lensing of the input mirrors and merges these with the CP-lenses
    into new CP lenses, and computes the new RoCs of the mirrors in the mirror list. Currently,
    only the test masses are supported.


    Input
    ------
    kat            - Kat-object to use for these computations.
    mirror_list    - List of test mass names to compute the thermal effects on.
    lensing        - If true, the input mirrors thermal lenses are computed.
    RoC            - If true, new RoCs for the mirrors are computed


    Returns
    -------

    new_params     - Dictionary with the new parameter values. E.g., {'CPN_TL': 1000, 'NI': -1700}
                     would mean that the focal length of the new compund CP+NI lens should be set
                     to 1000 m, and the new RoC of NI should be set to -1700 m.
    output         - Dictionary with a lot of auxiliary data from the process of computing the thermal
                     lenses, probably most important is the focal length of the input mirror lenses
                     at the input mirrors. 
    '''

    
    kat1 = kat.deepcopy()
    mirrors = kat1.IFO.mirrors
    cold_ifo = kat1.data['cold_optics_parameters']
    new_params = {}

    #################################
    # Compute new RoCs
    #################################
    
    if RoC:
        new_params.update(compute_thermal_RoCs(kat1, mirror_list))

    #################################
    # Compute thermal lensing
    #################################

    if lensing:
        
        # Get powers and spot sizes
        # -------------------------
        code = ""
        Ms = {}
        for m in mirror_list:
            # Getting HR-surface
            hr = kat1.components[m]

            # Getting AR-surface
            arname = m+'AR'
            if m == 'BS':
                arname += '1'
            ar = kat1.components[arname]

            # Getting substrate
            subname = 's'+m+'sub'
            if m == 'BS':
                subname += '1'
            sub = kat1.components[subname]

            # Storing compound mirrors in dictionary
            Ms[m] = {'HR': hr, 'SUB': sub, 'AR': ar}

            # Adding finesse-code for detecting power and spot sizes. Currently, this is only relevant for
            # input test masses, as theses are the only thermal lenses we currently care about. However,
            # code for the other mirrors is already included below. Note that the node definitions are
            # different depending on mirrors, therefore the if statement is here.
            if m == 'WI' or m == 'NI' or m == 'PR':
                # Power going into HR-side
                code += "pd P_{}_HR {}*\n".format(m, hr.nodes[1].name)
                # Substrate power, from AR to HR (or going into AR)
                code += "pd P_{}_sub1 {}\n".format(m, ar.nodes[1].name)
                # Substrate power, from HR to AR (or coming out of AR)
                code += "pd P_{}_sub2 {}*\n".format(m, ar.nodes[1].name)
                # Spot size
                code += "bp w_{}_x x w {}\n".format(m, hr.nodes[1].name)
                code += "bp w_{}_y y w {}\n".format(m, hr.nodes[1].name)
                # Complex beam parameter at the compensation plate. Used to compute errors when
                # merging the input mirror and CP lens into one new lens at the CP.
                if m == 'NI':
                    code += "bp q_{}_x x q {}\n".format("CPN", "nCPN_TL1")
                    code += "bp q_{}_y y q {}\n".format("CPN", "nCPN_TL1")
                elif m == 'WI':
                    code += "bp q_{}_x x q {}\n".format("CPW", "nCPW_TL1")
                    code += "bp q_{}_y y q {}\n".format("CPW", "nCPW_TL1")

            elif m == 'NE' or m == 'WE' or m == 'BS' or m == 'SR':
                # Power going into HR-side
                code += "pd P_{}_HR {}*\n".format(m, hr.nodes[0].name)
                # Substrate power, from AR to HR (or going into AR)
                code += "pd P_{}_sub1 {}\n".format(m, ar.nodes[0].name)
                # Substrate power, from HR to AR (or coming out of AR)
                code += "pd P_{}_sub2 {}*\n".format(m, ar.nodes[0].name)
                # Spot size
                code += "bp w_{}_x x w {}\n".format(m, hr.nodes[0].name)
                code += "bp w_{}_y y w {}\n".format(m, hr.nodes[0].name)

        code += 'noxaxis\n'
        code += 'yaxis abs:deg'

        kat1.parse(code)
        out = kat1.run()
        qN = (out['q_CPN_x'] + out['q_CPN_y'])/2.0
        qW = (out['q_CPW_x'] + out['q_CPW_y'])/2.0


        # Compute thermal lensing 
        # -------------------------
        # Dictionary for storing auxiliary data
        output = {}
        for k,v in Ms.items():
            # Computing thermal lens for input test masses
            if k == 'WI' or k == 'NI':
                # Preparing for computuing thermal lens
                # ------
                res = {}
                # Dictionary with mirror properties
                mp = copy.deepcopy(kat1.IFO.mirror_properties[k])

                mp['thickness'] = v['SUB'].L.value
                mp['n'] = v['SUB'].n.value
                mp['w'] = np.sqrt(out['w_{}_x'.format(k)].real)*np.sqrt(out['w_{}_y'.format(k)].real)

                mp['nScale'] = True

                P_coat = out['P_'+k+'_HR'].real
                P_sub_in = out['P_'+k+'_sub1'].real
                P_sub_out = out['P_'+k+'_sub2'].real

                # Comptuing the thermal lens
                res['f_thermal'], tmp = hellovinet(P_coat, P_sub_in, P_sub_out, mirror_properties = mp)
                res['r'] = tmp[0]
                res['OPL_data'] = tmp[1]

                # Combining CP and input mirror thermal lenses into one new lens at the CP
                if k == 'NI':
                    # Distance between CP and input mirror
                    d = kat1.sCPN_NI.L.value
                    new_params['CPN_TL'], errors = combine(cold_ifo['CPN_TL'], res['f_thermal'], d=d, q=qN)
                elif k == 'WI':
                    # Distance between CP and input mirror
                    d = kat1.sCPW_WI.L.value
                    new_params['CPW_TL'], errors = combine(cold_ifo['CPW_TL'], res['f_thermal'], d=d, q=qW)

                res['compound_lens_errs'] = errors
                output[k] = res

    return new_params, output


def compute_thermal_RoCs(kat, mirror_list):

    new_params = {}
    kat1 = kat.deepcopy()
    mirrors = kat1.IFO.mirrors
    cold_ifo = kat1.data['cold_optics_parameters']
    
    laser = kat1.getAll(pykat.components.laser)
    if len(laser) > 1:
        pkex.printWarning(("More than one laser is used. IFO.compute_thermal_effect() only "+
                           "gives correct results if the main laser is first in the tuple " +
                           "kat.getAll(pykat.components.laser)"))
    # Input laser power
    P_laser = laser[0].P.value

    for m in mirror_list:
        # Input mirrors
        if m == 'WI' or m == 'NI':
            a = -0.07506
        elif m == 'WE' or m == 'NE':
            a = 0.1004
        else:
            raise pkex.BasePyKatException(("Thermal RoC not supported for component {}. "+
                                           "Components must be test masses").format(m))
        
        # New RoC of HR-surface, formula from Valeria Sequino
        new_params[m] = a * P_laser + cold_ifo[m]

    return new_params
    



#def cavity_finesse_cmds(cavName, inputNodeName, cavNodeName):
#    return ("pd {0}_cav {1}\n"+
#            "pd {0}_in {2}\n"+
#            "noplot {0}_cav\n"+
#            "noplot {0}_in\n"+
#            "set {0}_c {0}_cav re\n"+
#            "set {0}_i {0}_in re\n"+
#            "func {0}_finesse = pi()*${0}_c/(2*${0}_i + 1E-21)\n"
#           ).format(cavName, cavNodeName, inputNodeName)

#def add_cavity_finesse_block(kat, cavs):
#    mirrors = kat.IFO.mirrors
#    cmd = ""
#    for cav in cavs:
#        if cav == 'N' or cav == 'X':
#            cmd += cavity_finesse_cmds(cav, "n{}1*".format(mirrors['IX']), "n{}2".format(mirrors['IX'])) 
#        elif cav == 'W' or cav == 'Y':
#            cmd += cavity_finesse_cmds(cav, "n{}1*".format(mirrors['IY']), "n{}2".format(mirrors['IY']))
#        elif cav == 'PRC':
#            cmd += cavity_finesse_cmds(cav, "n{}1*".format(mirrors['PRM']), 
#                                       kat.components[mirrors['BS']].nodes[0].name+'*')
#        else:
#            raise pkex.BasePyKatException("Cavity name {} is not supported. Must be X, Y, or PRC")
#    cmd += "yaxis lin abs\n"
#    kat.parse(cmd, addToBlock="cavityFinesse")
#    # print(cmd)
#    names = []
#    for c in cavs:
#        names.append(c+"_finesse")  
#    return names



def cavity_finesse_cmds(cavName, inputNodeName, cavNodeName, f = None):
    '''
    Returns Finesse code for measuring cavity finesse.
    
    Inputs
    ------
    cavName        - Name of cavity. Only used for naming. 
    inputNodeName  - Node name where to measure the input field.
    cavNodeName    - Node name where to measure the intra-cavity field
    f              - String with frequency name, thus, 'f1', 'f2', etc. If None or 0, 
                     the carrier frequency is used.
                     
    Returns
    -------
    cmd  - Finesse commands
    '''
    cmd = ""
    if f == 0 or f is None:
        f = 0
        cmd += ("ad {0}_cav_{3} {3} {1}\n"+
                "ad {0}_in_{3} {3} {2}\n")
    else:
        cmd += ("ad {0}_cav_{3} ${3} {1}\n"+
                "ad {0}_in_{3} ${3} {2}\n")
        
    cmd += ("noplot {0}_cav_{3}\n"+
            "noplot {0}_in_{3}\n"+
            "set {0}_c_{3} {0}_cav_{3} abs\n"+
            "set {0}_i_{3} {0}_in_{3} abs\n"+
            "func {0}_finesse_{3} = pi()*${0}_c_{3}*${0}_c_{3}/(2*${0}_i_{3}*${0}_i_{3} + 1E-21)\n")
    cmd = cmd.format(cavName, cavNodeName, inputNodeName, f)
    # print(cmd)
    return cmd

def add_cavity_finesse_block(kat, cavs, f=0):
    '''
    Adds finesse code for measuring the cavity finesse (=PowerGain*2/pi) at frequency f0 in
    one or several cavities. This function directly alters the kat-object. 
    
    Inputs
    ------
    kat   - kat-object. 
    cavs  - List with cavity names. Supported names: PRC, X or N, Y or W
    f     - String with frequency name, thus, 'f1', 'f2', etc. If None or 0, 
            the carrier frequency is used.
            
    Returns
    -------
    names - List with names of output signals.
    '''
    if f == 0 or f is None:
        f = 0
    elif not isinstance(f, str):
        raise pkex.BasePyKatException("f must be 0 or a string")
    elif not f in kat.constants.keys():
        raise pkex.BasePyKatException("f must be a frequeny in the kat-object")
                
    cmd = ""
    for cav in cavs:
        if cav == 'N' or cav == 'X':
            cmd += cavity_finesse_cmds(cav, "n{}1*".format('NI'), "n{}2".format('NI'), f=f) 
        elif cav == 'W' or cav == 'Y':
            cmd += cavity_finesse_cmds(cav, "n{}1*".format('WI'), "n{}2".format('WI'), f=f)
        elif cav == 'PRC':
            cmd += cavity_finesse_cmds(cav, "n{}1*".format('PR'), 
                                       kat.components['BS'].nodes[0].name+'*', f=f)
        else:
            raise pkex.BasePyKatException("Cavity name {} is not supported. Must be X, Y, or PRC")
    #cmd += "noxaxis\nyaxis lin abs\n"
    kat.parse(cmd, addToBlock="cavityFinesse")
    # print(cmd)
    names = []
    for c in cavs:
        names.append(c+"_finesse_{}".format(f))  
    return names



def rand_beta(kat, mirror_list, range):
    """
    Randomising mirror alignments uniformly between the values specifed in range. Mirror lists specifies which
    mirrors to do this for. Returns a new kat object with the random misalignments set. Thus, this function does
    not direclty alter the kat-object.
    """
    kat1 = kat.deepcopy()
    rand = np.random.rand(len(mirror_list)*2)
    rand = rand*(range[1]-range[0]) + range[0]
    k = 0
    for m in mirror_list:
        if (not m in kat.components or not (isinstance(kat.components[m], pykat.components.mirror) or
                                            isinstance(kat.components[m], pykat.components.beamSplitter))):
            raise pkex.BasePyKatException("{} is not a mirror or a beam splitterin the kat-object".format(m))
        kat1.components[m].xbeta = rand[k]
        kat1.components[m].ybeta = rand[k+1]
        k += 2
    return kat1
        

from blackbox import WingFFD
from baseclasses import AeroProblem
import numpy as np

class Wing():

    def __init__(self, seed=None):

        solverOptions = {
            # Common Parameters
            "monitorvariables": ["cl", "cd", "yplus"],
            "writeTecplotSurfaceSolution": True,
            "writeSurfaceSolution": False,
            "writeVolumeSolution": False,
            # Physics Parameters
            "equationType": "RANS",
            "smoother": "DADI",
            "MGCycle": "sg",
            "nsubiterturb": 10,
            "nCycles": 2000,
            # ANK Solver Parameters
            "useANKSolver": True,
            "ANKSubspaceSize": 400,
            "ANKASMOverlap": 3,
            "ANKPCILUFill": 4,
            "ANKJacobianLag": 5,
            "ANKOuterPreconIts": 3,
            "ANKInnerPreconIts": 3,
            # NK Solver Parameters
            "useNKSolver": True,
            "NKSwitchTol": 1e-6,
            "NKSubspaceSize": 400,
            "NKASMOverlap": 3,
            "NKPCILUFill": 4,
            "NKJacobianLag": 5,
            "NKOuterPreconIts": 3,
            "NKInnerPreconIts": 3,
            # Termination Criteria
            "L2Convergence": 1e-14
        }

        # Creating aeroproblem for adflow
        ap = AeroProblem(name="wing", alpha=2.5, mach=0.8, altitude=10000, areaRef=45.5, chordRef=3.56, evalFuncs=["cl", "cd", "cmy"])

        # Options
        options = {
            "solver": "adflow",
            "solverOptions": solverOptions,
            "gridFile": "wing_problem_files/wing_vol_mesh.cgns",
            "ffdFile": "wing_problem_files/ffd.xyz",
            "liftIndex": 3, # Very important
            "aeroProblem": ap,
            "noOfProcessors": 8,
            "sliceLocation": [0.14, 2.33, 4.67, 7.0, 9.33, 11.67, 13.86],
            "writeLiftDistribution": True,
            "writeDeformedFFD": True,
            "computeVolume": True,
            "leList": [[0.01, 0.001, 0.0], [7.51, 13.99, 0.0]],
            "teList": [[4.99, 0.001, 0.0], [8.99, 13.99, 0.0]],
            "samplingCriterion": "ese",
            "alpha": "implicit",
            "targetCL": 0.5,
            "targetCLTol": 1e-3,
            "startingAlpha": 2.5,
            "directory": "wing_case" if seed is None else f"wing_case_{seed}"
        }

        # Create the wing object
        self.wing = WingFFD(options=options)

        # Adding shape variables
        coeff = self.wing.DVGeo.origFFDCoef
        lidx = self.wing.DVGeo.getLocalIndex(0)
        fraction_local_thickness = 0.15

        # Twist
        self.wing.addDV("twist", lowerBound=np.array([-5.0]*self.wing.nTwist), upperBound=np.array([5.0]*self.wing.nTwist))

        # LE
        lowerBound = np.zeros(lidx.shape[1])
        upperBound = np.zeros(lidx.shape[1])

        for j in range(lidx.shape[1]): # loop over span
            local_thickness = coeff[lidx[0,j,1]][2] - coeff[lidx[0,j,0]][2]
            lowerBound[j] = -fraction_local_thickness*local_thickness
            upperBound[j] = fraction_local_thickness*local_thickness

        self.wing.addDV("shape_LE", lowerBound, upperBound)

        # TE
        lowerBound = np.zeros(lidx.shape[1])
        upperBound = np.zeros(lidx.shape[1])

        for j in range(lidx.shape[1]): # loop over span
            local_thickness = coeff[lidx[-1,j,1]][2] - coeff[lidx[-1,j,0]][2]
            lowerBound[j] = -fraction_local_thickness*local_thickness
            upperBound[j] = fraction_local_thickness*local_thickness

        self.wing.addDV("shape_TE", lowerBound, upperBound)

        # Remaining FFD variables
        lowerBound = np.zeros((lidx.shape[0]-2,lidx.shape[1],lidx.shape[2]))
        upperBound = np.zeros((lidx.shape[0]-2,lidx.shape[1],lidx.shape[2]))

        for i in range(lidx.shape[0]-2):

            for j in range(lidx.shape[1]):

                local_thickness = coeff[lidx[i+1,j,1]][2] - coeff[lidx[i+1,j,0]][2]

                upperBound[i,j,0] = fraction_local_thickness * local_thickness
                upperBound[i,j,1] = fraction_local_thickness * local_thickness

                lowerBound[i,j,0] = -fraction_local_thickness * local_thickness
                lowerBound[i,j,1] = -fraction_local_thickness * local_thickness

        self.wing.addDV("shape_local", lowerBound.flatten(), upperBound.flatten())

        self.lb = self.wing.lowerBound
        self.ub = self.wing.upperBound
        self.dim = self.wing.upperBound.shape[0]

    def __call__(self, x):

        output = self.wing.getObjectives(x.reshape(-1,))

        obj = output["cd"]

        if 1 - output["volume"] > 0:
            obj += 0.1*(1 - output["volume"])

        obj = np.array([obj])

        return obj
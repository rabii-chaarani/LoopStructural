import numpy as np

try:
    from . import DiscreteFoldInterpolator as DFI

    dfi = True
except ImportError:
    dfi = False
from . import FiniteDifferenceInterpolator as FDI

try:
    from . import PiecewiseLinearInterpolator as PLI

    pli = True
except ImportError:
    pli = False

# if LoopStructural.experimental:
from . import P2Interpolator

try:
    from . import SurfeRBFInterpolator as Surfe

    surfe = True

except ImportError:
    surfe = False

from . import StructuredGrid
from . import TetMesh

from ..modelling import BoundingBox
from ..utils import getLogger

logger = getLogger(__name__)


def get_interpolator(
    bounding_box: BoundingBox,
    interpolatortype="FDI",
    nelements=1e4,
    buffer=0.2,
    element_volume=None,
    **kwargs,
):
    """
    Returns an interpolator given the arguments, also constructs a
    support for a discrete interpolator

    Parameters
    ----------
    interpolatortype : string
        define the interpolator type
    nelements : int
        number of elements in the interpolator
    buffer : double or numpy array 3x1
        value(s) between 0,1 specifying the buffer around the bounding box
    data_bb : bool
        whether to use the model boundary or the boundary around
    kwargs : no kwargs used, this just catches any additional arguments

    Returns
    -------
    interpolator : GeologicalInterpolator
        A geological interpolator

    Notes
    -----
    This method will create a geological interpolator for the bounding box of the model. A
    buffer area is added to the interpolation region to avoid boundaries and issues with faults.
    This function wil create a :class:`LoopStructural.interpolators.GeologicalInterpolator` which can either be:
    A discrete interpolator :class:`LoopStructural.interpolators.DiscreteInterpolator`

    - 'FDI' :class:`LoopStructural.interpolators.FiniteDifferenceInterpolator`
    - 'PLI' :class:`LoopStructural.interpolators.PiecewiseLinearInterpolator`
    - 'P1'  :class:`LoopStructural.interpolators.P1Interpolator`
    - 'DFI' :class:`LoopStructural.interpolators.DiscreteFoldInterpolator`
    - 'P2'  :class:`LoopStructural.interpolators.P2Interpolator`
    or

    - 'surfe'  :class:`LoopStructural.interpolators.SurfeRBFInterpolator`

    The discrete interpolators will require a support.

    - 'PLI','DFI','P1Interpolator','P2Interpolator' :class:`LoopStructural.interpolators.supports.TetMesh` or you can provide another
        mesh builder which returns :class:`LoopStructural.interpolators.support.UnStructuredTetMesh`

    - 'FDI' :class:`LoopStructural.interpolators.supports.StructuredGrid`
    """
    bb = bounding_box.buffer(buffer)

    if interpolatortype == "PLI" and pli:
        if element_volume is None:
            # nelements /= 5
            element_volume = bb.volume / nelements
        # calculate the step vector of a regular cube
        step_vector = np.zeros(3)
        step_vector[:] = element_volume ** (1.0 / 3.0)
        # step_vector /= np.array([1,1,2])
        # number of steps is the length of the box / step vector
        nsteps = np.ceil((bb.length) / step_vector).astype(int)
        if np.any(np.less(nsteps, 3)):
            axis_labels = ["x", "y", "z"]
            for i in range(3):
                if nsteps[i] < 3:
                    logger.error(
                        f"Number of steps in direction {axis_labels[i]} is too small, try increasing nelements"
                    )
            logger.error("Cannot create interpolator: number of steps is too small")
            raise ValueError("Number of steps too small cannot create interpolator")
        # # create a structured grid using the origin and number of steps
        # if self.reuse_supports:
        #     mesh_id = f"mesh_{nelements}"
        #     mesh = self.support.get(
        #         mesh_id,
        #         TetMesh(origin=bb.origin, nsteps=nsteps, step_vector=step_vector),
        #     )
        #     if mesh_id not in self.support:
        #         self.support[mesh_id] = mesh
        else:
            if "meshbuilder" in kwargs:
                mesh = kwargs["meshbuilder"](bb, nelements)
            else:
                mesh = TetMesh(origin=bb.origin, nsteps=nsteps, step_vector=step_vector)
        logger.info(
            "Creating regular tetrahedron mesh with %i elements \n"
            "for modelling using PLI" % (mesh.ntetra)
        )

        return PLI(mesh)
    if interpolatortype == "P2":
        if element_volume is None:
            # nelements /= 5
            element_volume = bb.volume / nelements
        # calculate the step vector of a regular cube
        step_vector = np.zeros(3)
        step_vector[:] = element_volume ** (1.0 / 3.0)
        # step_vector /= np.array([1,1,2])
        # number of steps is the length of the box / step vector
        nsteps = np.ceil(bb.length / step_vector).astype(int)
        if "meshbuilder" in kwargs:
            mesh = kwargs["meshbuilder"](bb, nelements)
        else:
            raise NotImplementedError(
                "Cannot use P2 interpolator without external mesh"
            )
        logger.info(
            "Creating regular tetrahedron mesh with %i elements \n"
            "for modelling using P2" % (mesh.ntetra)
        )
        return P2Interpolator(mesh)
    if interpolatortype == "FDI":

        # find the volume of one element
        if element_volume is None:
            element_volume = bb.volume / nelements
        # calculate the step vector of a regular cube
        step_vector = np.zeros(3)
        step_vector[:] = element_volume ** (1.0 / 3.0)
        # number of steps is the length of the box / step vector
        nsteps = np.ceil((bb.length) / step_vector).astype(int)
        if np.any(np.less(nsteps, 3)):
            logger.error("Cannot create interpolator: number of steps is too small")
            axis_labels = ["x", "y", "z"]
            for i in range(3):
                if nsteps[i] < 3:
                    logger.error(
                        f"Number of steps in direction {axis_labels[i]} is too small, try increasing nelements"
                    )
            raise ValueError("Number of steps too small cannot create interpolator")
        # create a structured grid using the origin and number of steps
        # if self.reuse_supports:
        #     grid_id = "grid_{}".format(nelements)
        #     grid = self.support.get(
        #         grid_id,
        #         StructuredGrid(
        #             origin=bb.origin, nsteps=nsteps, step_vector=step_vector
        #         ),
        #     )
        #     if grid_id not in self.support:
        #         self.support[grid_id] = grid
        else:
            grid = StructuredGrid(
                origin=bb.origin, nsteps=nsteps, step_vector=step_vector
            )
        logger.info(
            f"Creating regular grid with {grid.n_elements} elements \n"
            "for modelling using FDI"
        )
        return FDI(grid)

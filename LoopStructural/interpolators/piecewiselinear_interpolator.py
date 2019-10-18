from LoopStructural.interpolators.discete_interpolator import DiscreteInterpolator
from LoopStructural.utils.helper import get_vectors
import numpy as np

import logging
logger = logging.getLogger(__name__)


class PiecewiseLinearInterpolator(DiscreteInterpolator):
    """
    Piecewise Linear Interpolator
    Approximates scalar field by finding coefficients to a piecewise linear
    equation on a tetrahedral mesh

    """

    def __init__(self, mesh):
        """

        Parameters
        ----------
        mesh
        """

        self.shape = 'rectangular'
        DiscreteInterpolator.__init__(self, mesh)
        # whether to assemble a rectangular matrix or a square matrix
        self.interpolator_type = 'PLI'
        self.nx = len(self.support.nodes[self.region])
        self.support = mesh

        self.interpolation_weights = {'cgw': 0.1, 'cpw' : 1., 'npw':1., 'gpw':1., 'tpw':1.}

    def copy(self):
        return PiecewiseLinearInterpolator(self.support)

    def _setup_interpolator(self, **kwargs):
        """
        adds all of the constraints to the interpolation matrix
        :param kwargs: 'cgw' is the constant gradient weight
        'cpw' control point weight
        'gpw' gradient control point weight
        'tpw' tangent control point weight
        'cg' boolean is cg being used
        :return:
        """
        # can't reset here, clears fold constraints
        #self.reset()
        for key in kwargs:
            self.up_to_date = False
            self.interpolation_weights[key] = kwargs[key]
        if self.interpolation_weights['cgw'] > 0.:
            self.up_to_date = False
            self.add_constant_gradient(self.interpolation_weights['cgw'])
        self.add_gradient_ctr_pts(self.interpolation_weights['gpw'])
        self.add_norm_ctr_pts(self.interpolation_weights['npw'])
        self.add_ctr_pts(self.interpolation_weights['cpw'])
        self.add_tangent_ctr_pts(self.interpolation_weights['tpw'])

    def add_constant_gradient(self, w=0.1):
        """
        Add the constant gradient regularisation to the system
        Parameters
        ----------
        w (double) - weighting of the cg parameter

        Returns
        -------

        """
        print("adding cg")
        # iterate over all elements
        A, idc, B = self.support.get_constant_gradient(region=self.region)
        A = np.array(A)
        B = np.array(B)
        idc = np.array(idc)

        gi = np.zeros(self.support.n_nodes)
        gi[:] = -1
        gi[self.region] = np.arange(0,self.nx)
        idc = gi[idc]
        # outside = ~np.any(idc == -1, axis=2)[:,0]
        # w/=A.shape[0]
        self.add_constraints_to_least_squares(A*w,B*w,idc)
        return

    def add_gradient_ctr_pts(self, w=1.0):
        """
        Adds gradient constraints to the least squares system with a weight defined by w
        Parameters
        ----------
        w - either numpy array of length number of

        Returns
        -------
        Notes
        -----
        Gradient constraints add a constraint that the gradient of the implicit function should
        be orthogonal to the strike vector and the dip vector defined by the normal.
        This does not control the direction of the gradient and therefore requires at least two other
        value constraints OR a norm constraint for the interpolant to solve.
        """
        points = self.get_gradient_constraints()
        if points.shape[0] > 0:
            e, inside = self.support.elements_for_array(points[:, :3])
            nodes = self.support.nodes[self.support.elements[e]]
            vecs = nodes[:,1:,:] - nodes[:,0,None,:]
            vol = np.abs(np.linalg.det(vecs)) #/ 6
            d_t = self.support.get_elements_gradients(e)
            norm = np.linalg.norm(d_t,axis=2)
            d_t /= norm[:,:,None]
            #d_t *= vol[:,None,None]
            # w*=10^11
            strike_vector, dip_vector = get_vectors(points[:,3:])
            A = np.einsum('ji,ijk->ik', strike_vector, d_t)

            A += np.einsum('ji,ijk->ik', dip_vector, d_t)
            A *= vol[:,None]

            #points[:,3:] /= norm

            # add in the element gradient matrix into the inte
            idc = self.support.elements[e]
            # now map the index from global to region create array size of mesh
            # initialise as np.nan, then map points inside region to 0->nx
            gi = np.zeros(self.support.n_nodes).astype(int)
            gi[:] = -1
            gi[self.region] = np.arange(0,self.nx).astype(int)
            w /= 3
            idc = gi[idc]
            B = np.zeros(idc.shape[0])
            outside = ~np.any(idc==-1,axis=1)
            self.add_constraints_to_least_squares(A[outside,:]*w,B[outside],idc[outside,:])

    def add_norm_ctr_pts(self, w=1.0):
        """
        Extracts the norm vectors from the interpolators p_n list and adds these to the implicit
        system
        Parameters
        ----------
        w

        Returns
        -------
        Notes
        -----
        Controls the direction and magnitude of the norm of the scalar field gradient.
        This constraint can conflict with value constraints if the magnitude of the vector doesn't
        match with the value constraints added to the implicit system.
        """

        points = self.get_norm_constraints()
        if points.shape[0] > 0:
            e, inside = self.support.elements_for_array(points[:, :3])
            nodes = self.support.nodes[self.support.elements[e]]
            vecs = nodes[:,1:,:] - nodes[:,0,None,:]
            vol = np.abs(np.linalg.det(vecs)) #/ 6
            d_t = self.support.get_elements_gradients(e)

            d_t *= vol[:,None,None]
            # w*=10^11

            points[:,3:] /= np.linalg.norm(points[:,3:],axis=1)[:,None]

            # add in the element gradient matrix into the inte
            e=np.tile(e,(3,1)).T
            idc = self.support.elements[e]
            w /= 3
            self.add_constraints_to_least_squares(d_t*w,points[:,3:]*w*vol[:,None],idc)
    def add_tangent_ctr_pts(self, w=1.0):
        """

        Parameters
        ----------
        w

        Returns
        -------

        """
        return

    def add_ctr_pts(self, w=1.0):  # for now weight all value points the same
        """
        Adds value constraints to the least squares system
        Parameters
        ----------
        w

        Returns
        -------

        """

        #get elements for points
        points = self.get_value_constraints()
        if points.shape[0] > 1:
            e, inside = self.support.elements_for_array(points[:, :3])
            # get barycentric coordinates for points
            nodes = self.support.nodes[self.support.elements[e]]
            vecs = nodes[:, 1:, :] - nodes[:, 0, None, :]
            vol = np.abs(np.linalg.det(vecs)) / 6
            A = self.support.calc_bary_c(e, points[:, :3])
            A *= vol[None,:]
            idc = self.support.elements[e]
            # now map the index from global to region create array size of mesh
            # initialise as np.nan, then map points inside region to 0->nx
            gi = np.zeros(self.support.n_nodes).astype(int)
            gi[:] = -1
            gi[self.region] = np.arange(0,self.nx)
            idc = gi[idc]
            outside = ~np.any(idc==-1,axis=1)
            self.add_constraints_to_least_squares(A[:,outside].T*w, points[outside, 3]*w*vol[None,outside], idc[outside,:])

    def add_gradient_orthogonal_constraint(self, elements, normals, w=1.0, B=0):
        """
        constraints scalar field to be orthogonal to a given vector
        Parameters
        ----------
        elements
        normals
        w
        B

        Returns
        -------

        """
        nodes = self.support.nodes[self.support.elements[elements]]
        vecs = nodes[:,1:,:] - nodes[:,0,None,:]
        vol = np.abs(np.linalg.det(vecs)) / 6
        d_t = self.support.get_elements_gradients(elements)
        dot_p = np.einsum('ij,ij->i', normals, normals)[:, None]
        mask = np.abs(dot_p) > 0
        normals[mask[:,0] ,:] =  normals[mask[:,0],:] / dot_p[mask][:,None]
        magnitude = np.einsum('ij,ij->i', normals, normals)
        normals[magnitude>0] = normals[magnitude>0] / magnitude[magnitude>0,None]
        A = np.einsum('ij,ijk->ik', normals, d_t)
        A *= vol[:,None]
        idc = self.support.elements[elements]
        B = np.zeros(len(elements))
        self.add_constraints_to_least_squares(A*w, B, idc)

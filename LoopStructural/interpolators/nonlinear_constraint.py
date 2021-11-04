import numpy as np
from .cython.dsi_helper import calculate_pairs
from scipy.sparse import coo_matrix

class BaseNonLinearConstraint:
    def __init__(self, feature):
        self.feature = feature
        pass

    @property
    def valid(self):
        return False

    def __call__(self,iteration):
        return 

class ConstantNormConstraint(BaseNonLinearConstraint):
    def __init__(self, feature, w):
        super().__init__( feature)
        self.w = w
    @property
    def valid(self):
        return True
    
    def __call__(self,w):
        pairs = np.array(calculate_pairs(self.feature.interpolator.support.get_neighbours(),self.feature.interpolator.support.get_elements()))
        pairs=pairs[~np.all(pairs==0,axis=1),:]
        # get the T matrix for all elements in the mesh
        element_gradients = self.feature.interpolator.support.get_element_gradients()
        bc = self.feature.interpolator.support.barycentre()
        # calculate gradient for every element for previous iteration
        vectors = self.feature.interpolator.support.evaluate_gradient(bc,self.feature.interpolator.c)
        # get vert indexes for all elements
        elements = self.feature.interpolator.support.get_elements()
        vertices = self.feature.interpolator.support.nodes[self.feature.interpolator.support.get_elements()]
        vecs = vertices[:, 1:, :] - vertices[:, 0, None, :]
        vol = np.abs(np.linalg.det(vecs))  # / 6
        # previous iteration gradient \cdot current iteration T1
        A1 = np.einsum('ij,ijk->ik', vectors[pairs[:,0],:], element_gradients[pairs[:,0],:,:])
        # weighting by volume was causing odd results
        A1 *= vol[pairs[:,0], None]
        # negative previous iteration gradient \cdot current iteration T2
        A2 = -np.einsum('ij,ijk->ik', vectors[pairs[:,1],:], element_gradients[pairs[:,1],:,:])
        A2 *= vol[pairs[:,1], None]
        # join A1,A2 
        a = np.hstack([A1,A2])
        idc = np.hstack([elements[pairs[:,0],:],elements[pairs[:,1],:]])
        B = np.zeros(a.shape[0])
        rows = np.tile(np.arange(a.shape[0]),(a.shape[1],1)).T
        A = coo_matrix((a.flatten(), (rows.flatten(), \
                                            idc.flatten())), shape=(a.shape[0], self.feature.interpolator.nx),
                        dtype=float)
        ATA = A.T.dot(A)
        ATB = A.T.dot(B)
        ATA*=self.w*w
        return ATA,ATB

class NormMagnitudeConstraint(BaseNonLinearConstraint):
    def __init__(self, feature, magnitude=None, step_calculator=lambda i: 10,w=1.):
        super().__init__(feature)
        self.magnitude = magnitude
        self.step_calculator = step_calculator
        self.w = w
    @property
    def valid(self):
        return True
    
    def __call__(self,w):
        bc = self.feature.interpolator.support.barycentre()
        # calculate gradient for every element for previous iteration
        vectors = self.feature.interpolator.support.evaluate_gradient(bc,self.feature.interpolator.c)
        if self.magnitude == None:
            self.magnitude = np.mean(np.linalg.norm(vectors,axis=1))
        vectors /= np.linalg.norm(vectors,axis=1)[:,None]
        vectors*=self.magnitude
        vertices, element_gradients, tetras, inside = self.feature.interpolator.support.get_element_gradient_for_location(bc)
        # e, inside = self.support.elements_for_array(points[:, :3])
        # nodes = self.support.nodes[self.support.elements[e]]
        vol = np.zeros(element_gradients.shape[0])
        vecs = vertices[:, 1:, :] - vertices[:, 0, None, :]
        vol = np.abs(np.linalg.det(vecs))  # / 6
        d_t = element_gradients
        d_t[inside,:,:] *= vol[inside, None, None]
        # add in the element gradient matrix into the inte
        idc = np.tile(tetras[:,:], (3, 1, 1))
        idc = idc.swapaxes(0,1)
        # idc = self.support.elements[e]
        gi = np.zeros(self.feature.interpolator.support.n_nodes).astype(int)
        gi[:] = -1
        gi[self.feature.interpolator.region] = np.arange(0, self.feature.interpolator.nx).astype(int)
        idc = gi[idc]
        outside = ~np.any(idc == -1, axis=2)
        outside = outside[:, 0]
        idc = idc.reshape((-1,idc.shape[2]))
        d_t = d_t.reshape((-1,d_t.shape[2]))
        rows = np.tile(np.arange(d_t.shape[0]),(d_t.shape[1],1)).T
        A = coo_matrix((d_t.flatten(), (rows.flatten(), \
                                                    idc.flatten())), shape=(d_t.shape[0], self.feature.interpolator.nx),
                                dtype=float)
        B = vectors.flatten()
        ATA = A.T.dot(A)
        ATB = A.T.dot(B)
        ATA*=w*self.w
        ATB*=self.w*w
            # w /= 3
        return ATA, ATB
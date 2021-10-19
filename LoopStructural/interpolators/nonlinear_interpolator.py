from LoopStructural.interpolators import DiscreteInterpolator
from LoopStructural.utils import LoopInterpolatorError

class NonLinearInterpolator(DiscreteInterpolator):
    def __init__(self, support):
        """
        """
        super().__init__(self, support)
        self._non_linear_constraints = {}

    def add_non_linear_constraint(self,constraint,name):
        if not issubclass(NonLinearConstraint,constraint) and constraint.valid:
            logger.error("{} is not a NonLinearConstraint".format(name))
            raise LoopInterpolatorError()
        self._non_linear_constraints[name] = constraint

    def _solve(self,solver,niter=10,loop_callback=None,**kwargs):
        logger.info("Iterative non linear solver: {} iterations".format(niter))
        A_base, B_base = feature.interpolator.build_matrix()
        self._solve(A_base,B_base,**kwargs)
        for i in range(niter):
            # make sure we don't modify the base matrices
            A = A_base.copy()
            B = B_base.copy()
            for constraint in self._non_linear_constraints:
                ATA, ATB = constraint(i)
                A+= ATA
                B+= ATB
            logger.info("Iteration: {}".format(i))
            self._solve(A,B,**kwargs)
            if callable(loop_callback):
                loop_callback(self)
        
    

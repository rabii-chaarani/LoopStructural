import numpy as np
class BoundingBox:
    def __init__(self, origin=None, maximum=None, minx=None, maxx=None, miny=None, maxy=None, minz=None, maxz=None):
        """A generic object for a models bounding box

        Parameters
        ----------
        origin : np.ndarray, optional
            lower left corner, by default None
        maximum : np.ndarray, optional
            upper right corner, by default None
        minx : float, optional
            minimum x coord, by default None
        maxx : float, optional
            max x coord, by default None
        miny : float, optional
            min y coord, by default None
        maxy : float, optional
            max y coord, by default None
        minz : float, optional
            min z coord, by default None
        maxz : float, optional
            max z coord, by default None
        """
        self._origin = None
        self._maximum = None
        if origin is not None:
            self.origin = origin
        if maximum is not None:
            self.maximum = maximum
        if minx is not None and miny is not None and minz is not None :
            self.origin = np.array([minx, miny, minz])
        if maxx is not None and maxy is not None and maxz is not None:
            self.maximum = np.array([maxx, maxy, maxz])
        self.is_valid()

    def __repr__(self) -> str:
        return f'BoundingBox(origin={self.origin}, maximum={self.maximum}) --> length={self.length}'
    @property
    def origin(self):
        return self._origin
    @origin.setter
    def origin(self, origin):
        origin = np.array(origin).astype(float)
        self._origin = origin
    @property
    def maximum(self):
        return self._maximum
    @maximum.setter
    def maximum(self, maximum):
        maximum = np.array(maximum).astype(float)
        self._maximum = maximum
    @property
    def length(self):
        if self.origin is not None and self.maximum is not None:
            return self.maximum - self.origin
        else:
            raise ValueError('Bounding box must have origin and maximum')
        
    def is_valid(self):
        if self.origin is None or self.maximum is None:
            raise ValueError('Bounding box must have origin and maximum')
        if not np.all(self.length>0):
            raise ValueError('Bounding box must have positive length')
        return True
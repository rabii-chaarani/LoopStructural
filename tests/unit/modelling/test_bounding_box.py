from LoopStructural.modelling import BoundingBox
import pytest
import numpy as np
@pytest.fixture(params=[0, 1, 2])
def bb_params(request):
    if request.param == 0:
        origin = np.zeros(3)
        maximm = np.ones(3)
        length = np.ones(3)
        success = True
    if request.param == 1:
        origin = np.zeros(3)
        maximm = np.ones(3)
        length = np.ones(3)
        success = True

    if request.param == 2:
        origin = np.zeros(3)
        maximm = -np.ones(3)
        length = np.ones(3)
        success = False
    return {'origin': origin, 'maximum': maximm, 'length': length, 'success': success}
def test_create_bounding_box(bb_params):
    success=True
    try:

        bb = BoundingBox(origin=bb_params['origin'], maximum=bb_params['maximum'])
        assert np.all(bb.origin == bb_params['origin'])
        assert np.all(bb.maximum == bb_params['maximum'])
        assert np.all(bb.length == bb_params['length'])
        assert bb.is_valid() == bb_params['success']
    except:
        success=False
    assert success == bb_params['success']
def test_create_bounding_box_from_min_max(bb_params):
    success=True
    try:

        bb = BoundingBox(minx=bb_params['origin'][0], miny=bb_params['origin'][1], minz=bb_params['origin'][2], maxx=bb_params['maximum'][0], maxy=bb_params['maximum'][1], maxz=bb_params['maximum'][2])
        assert np.all(bb.origin == bb_params['origin'])
        assert np.all(bb.maximum == bb_params['maximum'])
        assert np.all(bb.length == bb_params['length'])
    except Exception as e:
        success=False
        print(e)
    assert success == bb_params['success']

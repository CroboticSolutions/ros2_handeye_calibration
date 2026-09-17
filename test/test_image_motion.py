import numpy as np
from hand_eye_calibration.image_motion import ImageMotionModel


def test_unknown_joint_direction_is_not_extrapolated():
    m=ImageMotionModel(2);q=np.zeros(2);p=np.array([[.3,.3],[.7,.7]])
    m.add(q, np.array([.01,0]), p, p+[.01,0])
    assert m.predict(q,np.array([0,.02]),p) is None
    np.testing.assert_allclose(m.predict(q,np.array([.02,0]),p),p+[.02,0])


def test_large_prediction_error_discards_confidence_and_local_data():
    m=ImageMotionModel(2);p=np.array([[.3,.3],[.7,.7]])
    for i in range(5):
        m.add(np.array([i*.01,0]),np.array([(i+1)*.01,0]),p,p+[.01,0])
    assert m.good_predictions >= 3
    m.add(np.array([.05,0]),np.array([.06,0]),p,p+[.1,0])
    assert m.good_predictions==0 and len(m.samples)==1


def test_distant_configuration_requires_new_local_measurements():
    m=ImageMotionModel(2);p=np.array([[.3,.3],[.7,.7]])
    m.add(np.zeros(2), np.array([.01,0]),p,p+[.01,0])
    assert m.predict(np.ones(2),np.array([.01,0]),p) is None


def test_confidence_does_not_transfer_to_a_new_joint():
    m=ImageMotionModel(2);p=np.array([[.3,.3],[.7,.7]])
    for i in range(5):
        m.add(np.array([i*.01,0]),np.array([(i+1)*.01,0]),p,p+[.01,0])
    m.add(np.array([.05,0]),np.array([.05,.01]),p,p+[0,.01])
    assert m.confident(np.array([.05,.01]),np.array([.01,0]))
    assert not m.confident(np.array([.05,.01]),np.array([0,.01]))

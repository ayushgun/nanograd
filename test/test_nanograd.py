import math

import torch

from nanograd.grad import Scalar

tol = 2e-6


def test_cg():
    a = Scalar(2.0)
    b = Scalar(-4.0)
    c = a + b
    d = b * c
    e = -d
    f = e**3
    g = e / f
    h = g.relu()
    h.backward()

    ng_hd = h.data
    ng_ag = a.grad
    ng_bg = b.grad

    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(-4.0, requires_grad=True)
    c = a + b
    d = b * c
    e = -d
    f = e**3
    g = e / f
    h = torch.relu(g)
    h.backward()

    pt_hd = h.item()
    pt_ag = a.grad.item()
    pt_bg = b.grad.item()

    assert math.isclose(ng_hd, pt_hd)
    assert math.isclose(ng_ag, pt_ag)
    assert math.isclose(ng_bg, pt_bg)


test_cg()

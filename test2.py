import autodiff as ad
import numpy as np


x1 = ad.Variable(name = "x1")
x2 = ad.Variable(name = "x2")
x3 = ad.Variable(name = "x3")
y = x1 + x2 * x3 * x1
topo_order = ad.find_topo_sort([y])
reverse_topo_order = reversed(ad.find_topo_sort([y]))
for i in y.inputs:
    print(i)
grad_x1, grad_x2, grad_x3 = ad.gradients(y, [x1, x2, x3])

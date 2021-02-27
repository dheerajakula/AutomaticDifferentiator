import autodiff as ad
import numpy as np


x1 = ad.Variable(name = "x1")
x2 = ad.Variable(name = "x2")
x3 = ad.Variable(name = "x3")
x4 = ad.Variable(name = "x4")
y = x1 + x2 * x3 * x4

grad_x1, grad_x2, grad_x3, grad_x4 = ad.gradients(y, [x1, x2, x3, x4])

executor = ad.Executor([y, grad_x1, grad_x2, grad_x3, grad_x4])
topo_order = ad.find_topo_sort([y, grad_x1, grad_x2, grad_x3, grad_x4])
print(grad_x1, grad_x2, grad_x3, grad_x4)
for i in topo_order:
    print(i)
print("******")
x1_val = 1 * np.ones(3)
x2_val = 2 * np.ones(3)
x3_val = 3 * np.ones(3)
x4_val = 4 * np.ones(3)
y_val, grad_x1_val, grad_x2_val, grad_x3_val, grad_x4_val = executor.run(feed_dict = {x1 : x1_val, x2: x2_val, x3 : x3_val, x4 : x4_val})

print(grad_x2_val)
print(grad_x2)
for i in y.inputs:
    print(i)
assert isinstance(y, ad.Node)
assert np.array_equal(y_val, x1_val + x2_val * x3_val * x4_val)
assert np.array_equal(grad_x1_val, np.ones_like(x1_val))
assert np.array_equal(grad_x2_val, x3_val * x4_val)
assert np.array_equal(grad_x3_val, x2_val * x4_val)
assert np.array_equal(grad_x4_val, x2_val * x3_val)

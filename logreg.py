import autodiff as ad
import numpy as np

w = ad.Variable(name = "w")
x = ad.Variable(name = "x")
b = ad.Variable(name = "b")

labels = ad.Variable(name = "lables")

out = 1.0 / (1.0 + ad.exp_op((-1.0 * (w * x + b))))

ce_loss = -1.0 * ((labels * ad.log_op(out)) + ((1.0 - labels) * ad.log_op(1.0 - out)))

grad_w, grad_b = ad.gradients(ce_loss, [w,b])

# weights our model initially starts at
w_val = 10
b_val = 1

# weights our model should reach to 
w_required = 5
b_required = 20

# we are simulating the training dataset for logistic regression 

# taking x as a continuous array from -10 to 10 with step size of 0.01
x_val = np.arange(-10,6,0.01)

# finding the labels by doing the exact calculation of logistic regression using numpy
labels_val = 1/ (1+ np.exp(-( w_required * x_val + b_required )))

# logistic regression label values must be zero or one.
for i,t in np.ndenumerate(labels_val):
    if (t > 0.5):
        labels_val[i] = 1
    else:
        labels_val[i] = 0

executor = ad.Executor([out,ce_loss, grad_w, grad_b])

w_reached = 0
b_reached = 0

num_iterations = 10000
learning_rate = 1

for i in range(num_iterations):
    _,loss_value, grad_w_value, grad_b_value =  executor.run(feed_dict={x:x_val, w:w_val, b:b_val, labels:labels_val})
    w_val = w_val - learning_rate * np.mean(grad_w_value)
    b_val = b_val - learning_rate * np.mean(grad_b_value)
    if (i%10000 == 0):
        print(loss_value)
    w_reached = w_val
    b_reached = b_val


print(w_reached)
print(b_reached)

assert( w_required - 5< w_reached < w_required + 5)
assert( b_required - 5< b_reached < b_required + 5)
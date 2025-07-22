import numpy as np
import matplotlib.pyplot as plt


def loss(x, y, b):
    '''
    Compute loss given vectors x, y and scalar b
    '''

    return np.dot(y - b * x, y - b * x)

def gradient(x, y, b):
    '''
    Compute gradient given vectors x, y and scalar b
    '''

    return -2 * np.matmul((y - b*x).T, x)

def gradient_descent(x, y, b, e, epsilon=1e-6, max_itr=10000, verbose=True):
    '''
    Perform gradient descent with supplied input vectors x and y,
    an initial scalar value b, and a learning rate e. Algorithm
    terminates if the value of b converges or reaches the maximum
    number of iterations allowed.
    '''

    b_current = b
    n_itr = 1
    losses = []
    gradients = []

    for i in range(max_itr):
        # compute loss and gradient for current b
        losses.append(loss(x, y, b_current))
        gradients.append(gradient(x, y, b_current))
        
        if verbose:
            print(f"Starting {i}th iteration:\n loss={losses[i]}, current_b={b_current}")

        # update
        b_new = b_current - e * gradients[i]

        # check for convergence
        if abs(b_new - b_current) < epsilon:
            break

        b_current = b_new
        n_itr += 1

    return b_new, losses, gradients, n_itr


## set basic parameters ##
n = 10
lr = 1e-3
x = np.random.randn(n)
y = np.random.randn(n)
b_true = np.dot(x, y) / np.dot(x, x)
b_init = np.random.randint(0, 11)


## initial test ##
# run GD
b_est, loss_list, gradient_list, num_itr = gradient_descent(x=x, y=y, b=b_init, e=lr)
# diagnostics
print(f"x={x} and y={y}\n The true b={b_true} and the estimated b by gradient descent={b_est}")
# plotting
plt.figure()
plt.plot(range(len(loss_list)), loss_list)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.savefig('GD_initial_test.png')


## test with different lr ##
lr_list = np.logspace(-5, -1, num=10)
b_final_list = np.zeros(len(lr_list))
loss_final_list = np.zeros(len(lr_list))
itr_final_list = np.zeros(len(lr_list))

for j, lr in enumerate(lr_list):
    b_est, loss_list, gradient_list, num_itr = gradient_descent(x=x, y=y, b=b_init, e=lr, verbose=False)
    b_final_list[j] = b_est
    loss_final_list[j] = loss_list[-1]
    itr_final_list[j] = num_itr

# plot 1: loss versus learning rate
plt.figure()
plt.plot(lr_list, loss_final_list, marker='o')
plt.xlabel('Learning rates')
plt.ylabel('Final loss')
plt.title('Loss vs Learning Rate')
plt.savefig('GD_lr_v_loss.png')

# plot 2: number of iterations versus learning rate
plt.figure()
plt.plot(lr_list, itr_final_list, marker = 'o')
plt.xlabel('Learning rates')
plt.ylabel('# Iterations')
plt.title('Interations vs Learning Rate')
plt.savefig('GD_lr_v_itr.png')

# print out estimated b
print(lr_list)
print(b_final_list)
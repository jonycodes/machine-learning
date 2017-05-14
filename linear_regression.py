# Linear Regression with one feature.

from numpy import loadtxt, zeros, ones, array, linspace, logspace
from pylab import scatter, show, title, xlabel, ylabel, plot, contour  
data = loadtxt("ex1/ex1data1.txt", delimiter=",")

# print data
x = data[:, 0]
y = data[:, 1]

m = y.size 

scatter(x, y, marker='o', c='b')
title("Profit distribution in cities by population")
xlabel("Population is Cities, $10,000")
ylabel("Profit in $10,000")


# computing cost with mean of squares function
def compute_cost(X, y, theta):
    
    m = y.size
    
    # theta and x values are threated as vectors so we can use dot product to calculate t0 * x0 + t1 + x1, etc  
    predictions = X.dot(theta).flatten()

    # we square the error to normalize the data since we don't want negative values (mean of "squares")
    errors = (predictions - y) ** 2

    # Ordinary of two squares cost function or mean of square cost
    # we devide by m, which finds the mean of errors
    # we devived by 2 in order to make our gradient decent easier later on when we find the partial derivate
    j = (1.0 / (2 * m )) * errors.sum()

    return j

def gradient_decent(X, y, theta, alpha, num_iter):
    # data size
    m = y.size
    
    # matrix storing cost history
    J_history = zeros(shape=(num_iter, 1))

    for i in range(num_iter):

        # print X[:, 0]
        # print X[:, 1]
        
        # notice here how we are not using the compute_cost function and we are not squaring (prediction - y). 
        # Instead we multiply times the X value. This is because the gradient decent algorithm 
        # theta := theta - alpha * derivate of cost_function with respect to theta
        # Derivate of cost function with respect to theta0 and theta1:
        # theta0 yields 1/m * (theta + theta*x) 
        # theta1 yields 1/m * ((theta + theta*x) - y) * x 
        # this leaves two gradient decent functions 
        # theta0 := theta0 - alpha * 1/m * (theta*x - y) 
        # theta1 := theta1 - aplpha * 1/m * (theta*x - y) * x
        
        # first part of the gradient decent algorithm: make a prediction using the current theta0 and theta1 values 
        # theta0 and theta1 are first initiated to 0, but they could be initiated to any other number 
        prediction = X.dot(theta).flatten()
        
        # second part of gradient decent algorithm: find the difference between expected and actual value then multiply times the x value
        error_theta0 = (prediction - y)
        error_theta1 = (prediction - y) * X[:, 1]
        
        # third part of gradient decent: find the new theta simultaneously for theta0 and theta1 
        # using the rest of the gradient decent function theta := theta - (alpha * 1/m * errors) 
        # also since we're doing "batch" gradient decent, which means using all the training samples each iteration, we sum() the errors.
        theta[0][0] = theta[0][0] - alpha * ( 1.0 / m ) * error_theta0.sum() 
        theta[1][0] = theta[1][0] - alpha * ( 1.0 / m ) * error_theta1.sum()
        
        # compute the actual cost for analysis porpuses 
        J_history[i,  0] = compute_cost(X, y, theta)

    return theta, J_history




# h0 is the hypothesis function 
# The formula for our hypothesis function is h0 = t0*x0 + t1*x1, where t0 and t1 are the theta values
# store the x values on a vector  where x0 = 1 and x1 is the x value from the data
it = ones(shape=(m, 2))
it[:, 1] = x 

# x data points
# print it

# theta is a vector <t0, t1> initiated to 0 
theta = zeros(shape=(2, 1))

# print "First cost"
# print compute_cost(it, y, theta)

# standard learning rate
# our alpha step is constant because as the gradient approaches the local minimin, the gradient decent automatically takes
# smaller steps due to the property of the tangent (derivate of cost_function) and since the magnitute of the update 
# is proportional to our error, which means the less error the less the algorithm updates itself until eventually it reaches 0
alpha = 0.01
num_iter = 1500

theta, J_history =  gradient_decent(it, y, theta, alpha, num_iter)

print J_history
print theta.flatten()
results = it.dot(theta).flatten()
plot(data[:, 0], results)
show()



# Show contour graph
# Grid over which we will calculate 
theta0_vals = linspace(-10, 10, 100)
theta1_vals = linspace(-1, 4, 100)


#initialize J_vals to a matrix of 0's
J_vals = zeros(shape=(theta0_vals.size, theta1_vals.size))

#Fill out J_vals
for t1, element in enumerate(theta0_vals):
    for t2, element2 in enumerate(theta1_vals):
        thetaT = zeros(shape=(2, 1))
        thetaT[0][0] = element
        thetaT[1][0] = element2
        J_vals[t1, t2] = compute_cost(it, y, thetaT)

#Contour plot
J_vals = J_vals.T
#Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('theta_0')
ylabel('theta_1')
scatter(theta[0][0], theta[1][0])
show()

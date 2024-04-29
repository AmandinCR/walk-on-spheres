import numpy as np
import matplotlib.pyplot as plt

def cart_to_polar(x):
    r = np.linalg.norm(x)
    theta = np.arctan2(x[1],x[0]) # between -pi and pi
    if theta > 0:
        theta = theta - 2*np.pi # between 0 and -2pi
    return r,theta

def polar_to_cart(r,theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.array([x,y])

def true_solution(x):
    alpha = 1/3
    r,theta = cart_to_polar(x)
    return r**alpha * np.sin(alpha * theta) + np.exp(-r**2/2)

def f(x):
    r,theta = cart_to_polar(x)
    return (2-r**2) * np.exp(-r**2/2)

def get_distances(x):
    # this distnace function assumes x is never outside the domain
    dist_to_circle = np.abs(np.linalg.norm(x) - 1)
    
    # vertical side
    dist_to_side1 = 0
    if x[1] >= 0:
        dist_to_side1 = np.abs(x[0])
    else:
        dist_to_side1 = np.linalg.norm(x)
    
    # horizontal side
    dist_to_side2 = 0
    if x[0] >= 0:
        dist_to_side2 = np.abs(x[1])
    else:
        dist_to_side2 = np.linalg.norm(x)

    return dist_to_circle,dist_to_side1,dist_to_side2


def boundary_condition(x,epsilon):
    dist_to_circle,dist_to_side1,dist_to_side2 = get_distances(x)
    
    r,theta = cart_to_polar(x)
    if dist_to_circle <= epsilon:
        return np.sin(theta/3) + np.exp(-1/2)
    elif dist_to_side2 <= epsilon:
        return np.exp(-r**2/2)
    elif dist_to_side1 <= epsilon:
        return -r**(1/3) + np.exp(-r**2/2)
    else:
        print("ERROR: never met any boundary condition")
        print(x)
        exit()

def boundary_distance(x):
    dist_to_circle,dist_to_side1,dist_to_side2 = get_distances(x)
    dist_to_sides = np.minimum(dist_to_side1, dist_to_side2)
    return np.minimum(dist_to_circle,dist_to_sides)

def a(x):
    return boundary_distance(x)**2/4

def cdf(r,x0):
    d = boundary_distance(x0)
    return (2 * np.log(d / r) + 1) * r**2 / (2 * np.pi * d**2)

def inversion_sampling(x0,n=50):
    d = boundary_distance(x0)
    
    # 1. calculate the cum density
    # 2. invert the cum density numerically
    y = np.linspace(0,d,num=n)[1:] # r=0 will cause problems in cdf
    x = np.zeros(len(y))
    for i in range(len(y)):
        x[i] = cdf(y[i],x0)
    
    # 3. interpolate the inverted cum
    # 4. uniformly pick a value and plug into interpolation
    u = np.random.uniform(0,d)
    r = np.interp(u,x,y)
    return r

def point_in_sphere(x0):
    # need to generate r according to density rho
    theta = np.random.uniform(0,2*np.pi)
    r = inversion_sampling(x0)
    return np.array([x0[0] + r * np.cos(theta), 
                     x0[1] + r * np.sin(theta)])

def point_on_sphere(x0):
    r = boundary_distance(x0)
    x = np.random.normal(size=2)
    x = r * x/np.linalg.norm(x) + x0
    return x

def sample(x0,epsilon,max_iters=200):
    x = x0
    sol = 0
    j = 0
    while boundary_distance(x) > epsilon:
        y = point_in_sphere(x)
        sol += a(x)*f(y)
        x = point_on_sphere(x)

        if j > max_iters: 
            print("ERROR: never reached the boundary")
            exit()
            break
        else:
            j += 1
    
    sol += boundary_condition(x,epsilon)
    return sol, j

def main():
    epsilon = 5 * 10**(-5)
    #x0 = polar_to_cart(0.1244,-0.7906)
    #x0 = polar_to_cart(0.2320,-0.0274)
    #x0 = polar_to_cart(0.2187,-3.3975)
    x0 = polar_to_cart(0.1476,-4.1617)
    #x0 = polar_to_cart(0.0129,-1.4790)
    N = 500
    z = []
    num_steps = []
    for i in range(N):
        sol,j = sample(x0,epsilon)
        z.append(sol)
        num_steps.append(j)
    
    print("Monte Carlo = ", np.mean(z))
    print("Actual Solution = ", true_solution(x0))
    print("Error = ", np.abs(np.mean(z) - true_solution(x0)))
    print("Average number of steps = ", np.mean(num_steps))

main()    









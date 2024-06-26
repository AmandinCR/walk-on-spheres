import numpy as np
import matplotlib.pyplot as plt

def moving_average(x,w):
    return np.convolve(x,np.ones(w),'same')/w

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

def check_inside(x):
    if np.linalg.norm(x) > 1:
        return 0
    elif (x[1] > 0) and (x[0] > 0):
        return 0
    else:
        return 1

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
    x = [x0]
    y = []
    sol = 0
    j = 0
    while boundary_distance(x[j]) > epsilon:
        y.append(point_in_sphere(x[j]))
        sol += a(x[j])*f(y[j])
        x.append(point_on_sphere(x[j]))

        if j > max_iters: 
            print("ERROR: never reached the boundary")
            exit()
            break
        else:
            j += 1
    
    sol += boundary_condition(x[j],epsilon)
    x = np.array(x)
    y = np.array(y)
    return sol,x,y

def mcmc_solve(x0,epsilon,N):
    z = []
    num_steps = []
    for i in range(N):
        sol,x,y = sample(x0,epsilon)
        z.append(sol)
        num_steps.append(len(x))
    return z,num_steps

def plot_single_mc(x0,epsilon):
    sol,x,y = sample(x0,epsilon)
    print("Number of steps = ", len(x))
    
    plt.clf()
    plt.scatter(x[0,0],x[0,1],c="green")
    plt.scatter(x[1:-1,0],x[1:-1,1],c="blue")
    plt.scatter(x[-1,0],x[-1,1],c="red")
    # plot boundary
    t = np.linspace(np.pi/2,np.pi*2,100)
    plt.plot(np.cos(t),np.sin(t),linewidth=1,c="orange")
    plt.plot(np.zeros(10),np.linspace(0,1,10),linewidth=1,c="orange")
    plt.plot(np.linspace(0,1,10),np.zeros(10),linewidth=1,c="orange")
    plt.show()

def plot_epsilon_error(x0,N):
    epsilon_arr = np.linspace(0.001,0.1,10)
    acc_arr = np.zeros(len(epsilon_arr))
    var_arr = np.zeros(len(epsilon_arr))
    step_arr = np.zeros(len(epsilon_arr))
    for i in range(len(epsilon_arr)):
        z,num_steps = mcmc_solve(x0,epsilon_arr[i],N)
        acc_arr[i] = np.abs(np.mean(z) - true_solution(x0))
        var_arr[i] = np.var(z)
        step_arr[i] = np.mean(num_steps)
    plt.clf()
    plt.plot(epsilon_arr,acc_arr,'--bo')
    plt.show()

def plot_N_error(x0,epsilon):
    M = 5
    N_arr = np.arange(100,10000,100)
    avg_acc_arr = np.zeros(len(N_arr))
    for j in range(M):
        acc_arr = np.zeros(len(N_arr))
        z,_ = mcmc_solve(x0,epsilon,N_arr[-1])
        for i in range(len(N_arr)):
            acc_arr[i] = np.abs(np.mean(z[:N_arr[i]]) - true_solution(x0))
        # compute running average
        avg_acc_arr += acc_arr
    avg_acc_arr = avg_acc_arr / M
    plt.clf()
    plt.plot(N_arr,avg_acc_arr)
    plt.show()

def print_results(x0,epsilon,N):
    z,num_steps = mcmc_solve(x0,epsilon,N)
    print("Monte Carlo = ", np.mean(z))
    print("Actual Solution = ", true_solution(x0))
    print("Error = ", np.abs(np.mean(z) - true_solution(x0)))
    print("Average number of steps = ", np.mean(num_steps))

def plot_contour(epsilon,N):
    m = 15
    x = np.linspace(-1.0, 1.0,m)
    y = np.linspace(-1.0, 1.0,m)
    X,Y = np.meshgrid(x,y)
    Z = np.zeros((m,m))
    Z_true = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            x0 = np.array([X[i,j],Y[i,j]])
            if check_inside(x0):
                acc,_ = mcmc_solve(x0,epsilon,N)
                Z[i,j] = np.mean(acc)
                Z_true[i,j] = true_solution(x0)
                #Z[i,j] = np.abs(np.mean(acc) - true_solution(x0))
            else:
                Z[i,j] = -0.5
                Z_true[i,j] = -0.5
    plt.clf()
    plt.subplot(2,2,1)
    plt.title("walk-on-spheres solution")
    plt.contourf(X,Y,Z,cmap="bone")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")

    plt.subplot(2,2,2)
    plt.title("true solution")
    plt.contourf(X,Y,Z_true,cmap="bone")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("contour_wos.png",dpi=500,bbox_inches="tight")
    #plt.show()

def main():
    #x0 = polar_to_cart(0.1244,-0.7906)
    #x0 = polar_to_cart(0.2320,-0.0274)
    x0 = polar_to_cart(0.2187,-3.3975)
    #x0 = polar_to_cart(0.1476,-4.1617)
    #x0 = polar_to_cart(0.0129,-1.4790)
    epsilon = 0.001
    N = 100
    #print_results(x0,epsilon,N)
    #plot_single_mc(x0,epsilon)
    #plot_epsilon_error(x0,N)
    #plot_N_error(x0,epsilon)
    plot_contour(epsilon,N)
    
main()    









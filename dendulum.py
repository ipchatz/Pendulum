# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 18:13:06 2021

@author: yanni
"""

import numpy as np
from random import choice
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import odeint

m1, m2 = 0.048148, 0.044823 #bob masses (kg)
l1, l2 = 0.17, 0.17 #string lengths (m)
k1, k2 = 0, 0 #damping constants
M1, M2 = m1, m2 #rigid bob masses (kg)
s1, s2 = 0.13354, 0.13084042 #centres of mass (m)
I1, I2 = 0.00016697, 0.0001658 #moments of inertia (kgm^2)
c1, c2 = 0, 0 #damping on the hinges
a1, a2 = 0.001, 0 #damping on the centre of mass of each arm (air resistance)
b1, b2 = 0, 0 #damping on the bob mass (air resistance)
g = 9.807 #gravitational acceleration (ms^-2)
dt = 0.01 #time delta (s)
t_end = 10 #simulation time (s)
theta1_0 = 179 * np.pi / 180 #initial angle 1 (rad)
theta2_0 = -169 * np.pi / 180 #initial angle 2 (rad)
theta1 = theta1_0 #angular displacement 1 (rad)
theta2 = theta2_0 #angular displacement 2 (rad)
omega1_0 = 0 #intial angular velocity 1 (rads^-1)
omega2_0 = -1 #intial angular velocity 2 (rads^-1)
omega1 = omega1_0 #angular velocity 1 (rads^-1)
omega2 = omega2_0 #angular velocity 2 (rads^-1)
alpha1, alpha2 = 0, 0 #initial angular accelerations (rads^-2)
inits = [theta1_0, theta2_0, omega1_0, omega2_0] #intial conditions list
T = np.arange(0.0, t_end, dt) #an array to store the flow of time
s = lambda x : np.sin(x) #lambda function for sine
c = lambda x : np.cos(x) #lambda function for cosine

def genGraph(x, y, title_x = "", title_y = ""): #generates a specific graph
    if title_x != "": plt.xlabel(title_x) #adds an x-axis title if given
    if title_y != "": plt.ylabel(title_y) #adds a y=axis title if given
    plt.plot(x, y) #plots the dendulum position data
    plt.gca().set_aspect("equal", adjustable = "box") #sets unity aspect ratio
    plt.grid() #adds gridlines to the graph
    #plt.savefig("dendulum" + str(choice(T)) + ".png") #saves the figure
    plt.show() #shows the dendulum graph

def simDeudulum(init, t, l1, l2, m1, m2, g): #simulates the dendulum motion
    theta1, theta2, omega1, omega2 = init #initial state of the dendulum
    delta = theta1 - theta2 #difference between angular displacements
    M = m1 + m2 #total mass of the two bobs
    alpha1_a = m2 * g * s(theta2) * c(delta) #components of the equations of motion
    alpha1_b = - m2 * s(delta) * (l1 * (omega1**2) * c(delta) + l2 * (omega2**2))
    alpha1_c = - M * g * s(theta1)
    alpha1_d = l1 * (m1 + m2 * (s(delta))**2)
    alpha2_a = M * (l1 * (omega1**2) * s(delta))
    alpha2_b = M * g * (-s(theta2) + s(theta1) * c(delta))
    alpha2_c = m2 * l2 * (omega2**2) * s(delta) * c(delta)
    alpha2_d = l2 * (m1 + m2 * (s(delta))**2)
    alpha1 = (alpha1_a + alpha1_b + alpha1_c) / alpha1_d #equations of motion
    alpha2 = (alpha2_a + alpha2_b + alpha2_c) / alpha2_d
    cont = [omega1, omega2, alpha1, alpha2] #the derived state of the dendulum
    return cont #returns the first derivatives of the equations

def simDendulum(init, t, l1, l2, m1, m2, g): #simulates the dendulum
    theta1, theta2, omega1, omega2 = init #initial state of the dendulum
    delta = theta1 - theta2 #difference between angular displacements
    M = m1 + m2 #total mass of the two bobs
    alpha1_a = -g * (m1 + M) * s(theta1) #components of the differentials
    alpha1_b = - m2 * g * s(delta - theta2)
    alpha1_c = -2 * s(delta) * m2 * ((omega2**2) * l2 + (omega1**2) * l1 * c(delta))
    alpha1_d = l1 * (m1 + M + -m2 * c(2 * delta))
    alpha2_a = 2 * s(delta) * (omega1**2) * l1 * M
    alpha2_b = 2 * s(delta) * g * M * c(theta1)
    alpha2_c = 2 * s(delta) * (omega2**2) * l2 * m2 * c(delta)
    alpha2_d = l2 * (m1 + M + -m2 * (c(2 * delta))**2)
    alpha1 = (alpha1_a + alpha1_b + alpha1_c) / alpha1_d #equations of motion
    alpha2 = (alpha2_a + alpha2_b + alpha2_c) / alpha2_d
    cont = [omega1, omega2, alpha1, alpha2] #the derived state of the dendulum
    return cont #returns the first derivatives of the equations

def simChendulum(init, t, l1, l2, m1, m2, g): #simulates the dendulum motion
    theta1, theta2, omega1, omega2 = init #initial state of the dendulum
    delta = theta1 - theta2 #difference between angular displacements
    M = m1 + m2 #total mass of the two bobs
    alpha1_a = m2 * g * s(theta2) * c(delta) #components of the differentials
    alpha1_b = -m2 * l2 * (omega1**2) * s(delta) * c(delta)
    alpha1_c = -m2 * l2 * (omega2**2) * s(delta) - g * M * s(theta1)
    alpha1_d = M * l1 - m2 * l1 * (c(delta))**2
    alpha2_a = m2 * l2 * (omega1**2) * s(delta)
    alpha2_b = -m2 * g * s(theta2)
    alpha2_c = (m2 / M) * (m2 * l2 * (omega1**2) * s(delta) + g * M * s(theta1)) / c(delta)
    alpha2_d = m2 * l2 - (m2**2) * l2 * ((c(delta))**2) / M
    alpha1 = (alpha1_a + alpha1_b + alpha1_c) / alpha1_d #equations of motion
    alpha2 = (alpha2_a + alpha2_b + alpha2_c) / alpha2_d
    cont = [omega1, omega2, alpha1, alpha2] #the derived state of the dendulum
    return cont #returns the first derivatives of the equations

def simDDendulum(init, t, l1, l2, m1, m2, g, k1, k2): #simulates the dendulum motion
    theta1, theta2, omega1, omega2 = init #initial state of the dendulum
    delta = theta1 - theta2 #difference between angular displacements
    M = m1 + m2 #total mass of the two bobs
    alpha1_a = m2 * l1 * (omega1**2) * s(2 * delta) #equation components
    alpha1_b = 2 * m2 * l2 * (omega2**2) * s(delta)
    alpha1_c = 2 * m2 * g * c(theta2) * s(delta) + 2 * m1 * g * s(theta1)
    alpha1_d = 2 * k1 * omega1 - 2 * k2 * omega2 * c(delta)
    alpha1_e = -2 * l1 * (m1 + m2 * (s(delta))**2)
    alpha2_a = m2 * l2 * (omega2**2) * s(2 * delta)
    alpha2_b = 2 * M * l1 * (omega1**2) * s(delta)
    alpha2_c = 2 * M * g * c(theta1) * s(delta)
    alpha2_d = 2 * k1 * (omega1) * c(delta)
    alpha2_e = -2 * M * k2 * omega2 / m2
    alpha2_f = 2 * l2 * (m1 + m2 * (s(delta))**2)
    alpha1 = (alpha1_a + alpha1_b + alpha1_c + alpha1_d) / alpha1_e #equations
    alpha2 = (alpha2_a + alpha2_b + alpha2_c + alpha2_d + alpha2_e) / alpha2_f
    cont = [omega1, omega2, alpha1, alpha2] #the derived state of the dendulum
    return cont #returns the first derivatives of the equations

def simDrigilum(init,t,l1,l2,M1,M2,g,s1,s2,I1,I2): #eqns of motion for rigid compound dendulum
    theta1, theta2, omega1, omega2 = init #initial state of the dendulum
    delta = theta1 - theta2 #difference between angular displacements
    an1=M2*g*s2*s(theta2)-M2*l1*omega1**2*s2*s(delta)
    afac1=M2*l1*s2*c(delta)/(M2*s2**2+I2)
    an2=-M2*l1*s2*omega2**2*s(delta)
    an3=-(M1*g*s1+M2*g*l1)*s(theta1)
    ad1=M1*s1**2+I1+M2*l1**2
    ad2=-(M2*l1*s2*c(delta)**2*M2*l1*s2)/(M2*s2**2+I2)
    alpha1 = (an1*afac1+an2+an3)/(ad1+ad2) #
    bn1=M2*g*s2*s(theta2)-M2*l1*omega1**2*s2*s(delta)
    bfac1=(M1*s1**2+I1+M2*l1**2)/(M2*l1*s2*c(delta))
    bn2=-M2*l1*s2*omega2**2*s(delta)
    bn3=-(M1*g*s1+M2*g*l1)*s(theta1)
    bd1=M2*l1*s2*c(delta)
    bd2=-(M2*s2**2+I2)*(M1*s1**2+I1+M2*l1**2)/(M2*l1*s2*c(delta))
    alpha2 =(bn1*bfac1+bn2+bn3)/(bd1+bd2)
    cont = [omega1, omega2, alpha1, alpha2] #the derived state of the dendulum
    return cont #returns the first derivatives of the equations
 
def simDDrigilum(init,t,l1,l2,M1,M2,g,s1,s2,I1,I2,c1,c2,a1,a2,b1,b2):  #eqns of motion for rigid compound dendulum
    theta1, theta2, omega1, omega2 = init #initial state of the dendulum
    delta = theta1 - theta2 #difference between angular displacements
    if [np.absolute(omega1),np.absolute(omega2)] < [0.000001,0.000001]: Gamma1,Gamma2=0,0
    else: Gamma1,Gamma2=c1,c2
    G1=-Gamma1-omega1*(a1*s1+b1*l1)  #total toruqe on arm1, Gamma is the constant toque due to the bearings (very small),alpha is the  damp constant for the plastic part of the arm, beta is the damping due to bolts/blu-tack 
    G2=-Gamma2-omega2*(a2*s2+b2*l2)  # total toque on arm2
    an1=M2*g*s2*s(theta2)-M2*l1*omega1**2*s2*s(delta)-G2
    afac1=M2*l1*s2*c(delta)/(M2*s2**2+I2)
    an2=-M2*l1*s2*omega2**2*s(delta)
    an3=-(M1*g*s1+M2*g*l1)*s(theta1)+G1
    ad1=M1*s1**2+I1+M2*l1**2
    ad2=-(M2*l1*s2*c(delta)**2*M2*l1*s2)/(M2*s2**2+I2)
    eqA=(an1*afac1+an2+an3)/(ad1+ad2)
    bn1=M2*g*s2*s(theta2)-M2*l1*omega1**2*s2*s(delta)-G2
    bfac1=(M1*s1**2+I1+M2*l1**2)/(M2*l1*s2*c(delta))
    bn2=-M2*l1*s2*omega2**2*s(delta)
    bn3=-(M1*g*s1+M2*g*l1)*s(theta1)+G1
    bd1=M2*l1*s2*c(delta)
    bd2=-(M2*s2**2+I2)*(M1*s1**2+I1+M2*l1**2)/(M2*l1*s2*c(delta))
    eqB=(bn1*bfac1+bn2+bn3)/(bd1+bd2)
    cont = [omega1, omega2, eqA, eqB] #the derived state of the dendulum
    return cont #returns the first derivatives of the equations

def main(simFunction): #main function to run the simulation
    if simFunction == simDDendulum: #checks if the sim function is damped
        sol = odeint(simDDendulum, inits, T, (l1, l2, m1, m2, g, k1, k2))

    elif simFunction == simDDrigilum: #if the sim function is rigid damped
        sol = odeint(simDDrigilum, inits, T, (l1,l2,M1,M2,g,s1,s2,I1,I2,c1,c2,a1,a2,b1,b2))
 
    elif simFunction == simDrigilum: #if the sim function is compound rigid
        sol = odeint(simDrigilum, inits, T, (l1,l2,M1,M2,g,s1,s2,I1,I2))

    else: #otherwise run the standard simulation without damping
        sol = odeint(simFunction, inits, T, (l1, l2, m1, m2, g))

    theta1 = sol[:, 0] #obtains the theta 1 values from the solution
    theta2 = sol[:, 1] #obtains the theta 2 values from the solution

    x1 = l1 * np.sin(theta1) #obtains the x1 positions from the solution
    y1 = -l1 * np.cos(theta1) #obtains the y1 positions from the solution
    x2 = x1 + l2 * np.sin(theta2) #obtains the x2 positions from the solution
    y2 = y1 - l2 * np.cos(theta2) #obtains the y2 positions from the solution

    plt.plot(x2, y2) #plot the position of the outer pendulum bob
    genGraph(x1, y1, "x", "y") #plots the dendulum position graph

sol = odeint(simDDendulum, inits, T, (l1, l2, m1, m2, g, k1, k2)) #solve

theta1 = sol[:, 0] #obtains the theta values from the solution
theta2 = sol[:, 1] #obtains the theta values from the solution
omega1 = sol[:, 2] #obtains the omega values from the solution
omega2 = sol[:, 3] #obtains the omega values from the solution

#genGraph(T, theta1, "t", "theta1") #plots the angle against time graph
#genGraph(T, theta2, "t", "theta2") #plots the angle against time graph
#genGraph(T, omega1, "t", "omega1") #plots the angular velocity graph
#genGraph(T, omega2, "t", "omega2") #plots the angular velocity graph

x1 = l1 * np.sin(theta1) #obtains the x1 positions from the solution
y1 = -l1 * np.cos(theta1) #obtains the y1 positions from the solution
x2 = x1 + l2 * np.sin(theta2) #obtains the x2 positions from the solution
y2 = y1 - l2 * np.cos(theta2) #obtains the y2 positions from the solution

#genGraph(x1, y1, "x1", "y1") #plots the dendulum position graph
#genGraph(T, x1, "t", "x1") #plots the x positions against time graph
#genGraph(T, y1, "t", "y1") #plots the height against time graphs
#genGraph(x2, y2, "x2", "y2") #plots the dendulum position graph
#genGraph(T, x2, "t", "x2") #plots the x positions against time graph
#genGraph(T, y2, "t", "y2") #plots the height against time graphs

#plt.plot(x2, y2) #plot sthe position of the outer pendulum bob
#genGraph(x1, y1, "x", "y") #plots the dendulum position graph

for n in range(len(theta1)): #iterates through the inner theta values
    if np.absolute(theta1[n]) > np.pi: #checks if the value exceeds pi
        print(n, theta1[n], T[n]) #prints the flip values
        break #stops iterating because the first flip has been found

for n in range(len(theta2)): #iterates through the outer theta values
    if np.absolute(theta2[n]) > np.pi: #checks if a flip has occurred
        print(n, theta2[n], T[n]) #outputs the flip values
        break #stops the iteration since the first flip has been found

funcs = [simDeudulum, simDDendulum, simDrigilum, simDDrigilum] #simu functions
for simFunc in funcs: #iterates through each of the functions in the list
    main(simFunc) #runs each simulation function for the initial values
'''
thetas1 = thetas2 = np.linspace(0, np.pi, 180) #range of theta values
tthetas1, tthetas2 = [], [] #lists to store the theta flip values
flip_times1, flip_times2 = [], [] #lists to store the times taken to flip
for t1 in thetas1: #iterates through all of the theta 1 values
    for t2 in thetas2: #iterates through all of the theta 1 values
        n1, n2 = -1, -1 #counters for the while loop
        inits = [t1, t2, omega1_0, omega2_0] #intial conditions list
        sol = odeint(simDeudulum, inits, T, (l1, l2, m1, m2, g)) #solves
        theta1 = sol[:, 0] #obtains the theta 1 values from the solution
        theta2 = sol[:, 1] #obtains the theta 2 values from the solution
        flip_flag1, flip_flag2 = False, False #flags to store the states
        while n1 < len(theta1) - 1 and flip_flag1 == False: #iterates through thetas
            n1 += 1 #increments the counter
            if np.absolute(theta1[n1]) > np.pi: #checks if the value exceeds pi
                flip_times1.append(T[n1]) #prints the time taken to flip value
                tthetas1.append(t1) #appends the flip angle
                flip_flag1 = True #stops the iteration as found first flip

        while n2 < len(theta2) -1 and flip_flag2 == False: #iterates through thetas
            n2 += 1 #increments the counter
            if np.absolute(theta2[n2]) > np.pi: #checks if a flip has occurred
                flip_times2.append(T[n2]) #appends the time taken to flip value
                tthetas2.append(t2) #appends the flip angle
                flip_flag2 = True #stops the iteration as found first flip

plt.scatter(tthetas1, flip_times1, marker=".") #plots the dendulum flip 1 graph
plt.grid() #adds gridlines to the graph
plt.xlabel("Initial bob 1 angle") #adds an x-axis title
plt.ylabel("Time to flip") #adds a y-axis title
plt.savefig("ddendulum" + str(choice(flip_times1)) + ".png") #saves the figure
plt.show() #shows the graph
plt.scatter(tthetas2, flip_times2, marker=".") #plots the dendulum flip 2 graph
plt.grid() #adds gridlines to the graph
plt.xlabel("Initial bob 2 angle") #adds an x-axis title
plt.ylabel("Time to flip") #adds an y-axis title
plt.savefig("ddendulum" + str(choice(flip_times2)) + ".png") #saves the figure
plt.show() #shows the graph
'''
size_fig = l1 + l2 #assigns the value of the total length to the figure size
size_trail = 21 #assigns the value to the trail size
fig = plt.figure() #creates a figure to plot the graph animation
ax = fig.add_subplot(111, autoscale_on = False, xlim=(-size_fig, size_fig), ylim = (-size_fig, size_fig))
ax.grid() #adds gridlines, x-axis and y-axis limits
ax.set_aspect("equal", adjustable = "box") #sets the axes to unity
line, = ax.plot([], [], "o-", lw = 2) #creates the double line variable
trail, = ax.plot([], [], ",--", lw = 1) #creates the trail line variable
trail_x, trail_y = [None]*size_trail, [None]*size_trail #trail lines lists

def init(): #initialisation function for the background of each frame
    line.set_data([], []) #initialises the dendulum simulation data
    trail.set_data([], []) #initialises the dendulum trail data
    return line, trail, #time_text #returns the initial background

def animate(i): #animation function for each individual frame
    global trail_x, trail_y #declares the lists as global variables
    line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]]) #sets the frame data
    if i == 0: trail_x, trail_y = [None]*size_trail, [None]*size_trail #reset
    trail_x.append(x2[i]) #appends the new frame data to the x2 trail list
    trail_y.append(y2[i]) #appends the new frame data to the y2 trail list
    while len(trail_x) > size_trail: trail_x.pop(0) #removes the first element
    while len(trail_y) > size_trail: trail_y.pop(0) #if trail list exceeds max
    trail.set_data(trail_x, trail_y) #sets the trail frame data
    return line, trail, #time_text #returns the updated frame data

anim = animation.FuncAnimation(fig, animate, int(t_end/dt), init, blit=True)
anim.save("dandulum" + str(choice(x2)) + ".gif", fps=30) #saves the gif
plt.show() #outputs the animated graph of the dendulum simulation

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

m1, m2 = 1, 1 #bob masses (kg)
l1, l2 = 1, 1 #string lengths (m)
k1, k2 = 0, 0 #damping constants
g = 9.807 #gravitational acceleration (ms^-2)
dt = 0.01 #time delta (s)
t_end = 5 #simulation time (s)
theta1_0 = 179 * np.pi / 180 #initial angle 1 (rad)
theta2_0 = -169 * np.pi / 180 #initial angle 2 (rad)
theta1 = theta1_0 #angular displacement 1 (rad)
theta2 = theta2_0 #angular displacement 2 (rad)
omega1_0 = 0 #intial angular velocity 1 (rads^-1)
omega2_0 = 0 #intial angular velocity 2 (rads^-1)
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

def main(simFunction): #main function to run the simulation
    if simFunction == simDDendulum: #checks if the sim function is damped
        sol = odeint(simDDendulum, inits, T, (l1, l2, m1, m2, g, k1, k2))

    else: #otherwise run the standard simulation without damping
        sol = odeint(simFunction, inits, T, (l1, l2, m1, m2, g))

    theta1 = sol[:, 0] #obtains the theta 1 values from the solution
    theta2 = sol[:, 1] #obtains the theta 2 values from the solution
    omega1 = sol[:, 2] #obtains the omega 1 values from the solution
    omega2 = sol[:, 3] #obtains the omega 2 values from the solution

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

funcs = [simDeudulum, simDDendulum] #a list to store the simulation functions
for simFunc in funcs: #iterates through each of the functions in the list
    main(simFunc) #runs each simulation function for the initial values

fig = plt.figure() #creates a figure to plot the graph animation
ax = fig.add_subplot(111, autoscale_on = False, xlim=(-2, 2), ylim = (-2, 2))
ax.grid() #adds gridlines, x-axis and y-axis limits
ax.set_aspect("equal", adjustable = "box") #sets the axes to unity
line, = ax.plot([], [], "o-", lw = 2) #creates the double line variable

def init(): #initialisation function for the background of each frame
    line.set_data([], []) #initialises the dendulum simulation data
    return line, #returns the initial background

def animate(i): #animation function for each individual frame
    line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]]) #sets the frame data
    return line, #returns the updated frame data 

anim = animation.FuncAnimation(fig, animate, int(t_end/dt), init, blit=True)
#anim.save("dandulum" + str(choice(T)) + ".gif", fps=30) #saves the gif
plt.show() #outputs the animated graph of the dendulum simulation

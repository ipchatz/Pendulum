# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 18:10:02 2021

@author: yanni
"""

import numpy as np
from random import choice
import matplotlib.pyplot as plt
from scipy.integrate import odeint

m = 1 #bob mass (kg)
l = 1 #string length (m)
g = 9.807 #gravitational acceleration (ms^-2)
b = 0.42 #damping const
dt = 0.01 #time delta (s)
t_end = 5 #simulation time (s)
theta_0 = np.pi / 6 #initial angle (rad)
theta = theta_0 #angular displacement (rad)
omega_0 = 0 #intial angular velocity (rads^-1)
omega = omega_0 #angular velocity (rads^-1)
alpha = 0 #initial angular acceleration (rads^-2)
inits = [theta_0, omega_0] #a list to store the intial conditions
T = np.arange(0, t_end, dt) #an array to store the flow of time

def genGraph(x, y, title_x = "", title_y = ""): #generates a specific graph
    if title_x != "": plt.xlabel(title_x) #adds an x-axis title if given
    if title_y != "": plt.ylabel(title_y) #adds a y=axis title if given
    plt.plot(x, y) #plots the pendulum position data
    #plt.gca().set_aspect("equal") #sets the aspect ratio to unity
    plt.grid() #adds gridlines to the graph
    plt.savefig("pendulum" + str(choice(T)) + ".png") #saves the figure
    plt.show() #shows the pendulum graph

def simPendulum(init, t, l, m, b, g): #simulates the pendulum motion
    theta, omega = init #initial state of the pendulum
    alpha = -b / m * omega - g / l * np.sin(theta) #equation of motion
    cont = [omega, alpha] #the derived state of the position of the pendulum
    return cont #returns the first derivative of the equation


####################   Differential Pendulum Simulation   ####################
sol = odeint(simPendulum, inits, T, (l, m, b, g)) #solve the differential

theta = sol[:, 0] #obtains the theta values from the solution
omega = sol[:, 1] #obtains the omega values from the solution

genGraph(T, theta, "t", "theta") #plots the angle against time graph
genGraph(T, omega, "t", "omega") #plots the angular velocity graph

x = l * np.sin(theta) #obtains the x positions from the solution
y = -l * np.cos(theta) #obtains the y positions from the solution

genGraph(x, y, "x", "y") #plots the pendulum position graph
genGraph(T, x, "t", "x") #plots the x positions against time graph
genGraph(T, y, "t", "y") #plots the height against time graphs


''' ####################   SHM Simulation   ####################
l = 1 #string length (m)
g = 9.807 #gravitational acceleration (ms^-2)
dt = 0.01 #time delta (s)
t_end = 1 #simulation time (s)
theta_0 = np.pi / 6 #initial angle (rad)
theta = theta_0 #angular displacement (rad)
omega = np.sqrt(g / l) #intial angular velocity (rads^-1)
T = np.arange(0, t_end, dt) #an array to store the flow of time
x_positions = [] #a list to store the x-path of the pendulum bob
y_positions = [] #a list to store the y-path of the pendulum bob

for t in T: #iterates through all of the time-steps
    x = l * np.sin(theta) #obtains the x-position at a given time
    y = l * np.cos(theta) #obtains the height at a given time

    theta = theta_0 * np.cos(omega * t) #iterates the angle to the next time

    x_positions.append(x) #appends the x-position to the specific list
    y_positions.append(-y) #appends the absolute value of the height

genGraph(x_positions, y_positions, "x", "y") #plots the pendulum position
genGraph(T, x_positions, "t", "x") #plots the x-position against time
genGraph(T, y_positions, "t", "y") #plots the height against time graph
'''

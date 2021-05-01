# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 18:10:02 2021

@author: yanni
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

m = 1 #bob mass (kg)
l = 1 #string length (m)
g = 9.807 #gravitational acceleration (ms^-2)
b = 0 #damping const
dt = 0.01 #time delta (s)
t_end = 1 #simulation time (s)
theta_0 = np.pi / 6 #initial angle (rad)
theta = theta_0 #angular displacement (rad)
omega = 0 #intial angular velocity (rads^-1)
alpha = 0 #initial angular acceleration (rads^-2)
T = np.arange(0, t_end, dt) #an array to store the flow of time
x_positions = [] #a list to store the x-path of the pendulum bob
y_positions = [] #a list to store the y-path of the pendulum bob

def genGraph(x, y, title_x = "", title_y = ""): #generates a specified graph
    if title_x != "": plt.xlabel(title_x) #adds an x-axis title if given
    if title_y != "": plt.ylabel(title_y) #adds a y=axis title if given
    plt.plot(x, y) #plots the data
    plt.gca().set_aspect("equal") #sets the aspect ratio to unity
    plt.show() #shows the graph

def simPendulum(theta, omega, t, l, m, b, g): #simulates the pendulum motion
    alpha = -b / m * omega - g / l * np.sin(theta) #equation of motion
    return alpha #angular acceleration as a function of time
'''
for t in T:
    x = l * np.sin(theta)
    y = l * np.cos(theta)

    alpha = simPendulum(theta, omega, t, l, m, b, g)
    omega += alpha
    theta += omega

    x_positions.append(x)
    y_positions.append(y)

genGraph(x_positions, y_positions, "x", "y")

'''

omega = (g / l) ** 0.5 #small angle approximation

for t in T:
    x = l * np.sin(theta)
    y = l * np.cos(theta)

    theta = theta_0 * np.cos(omega * t)

    x_positions.append(x)
    y_positions.append(-y)

genGraph(x_positions, y_positions, "x", "y")

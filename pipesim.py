import numpy as np
import matplotlib.pyplot as plt

R = 2.5                  # Tube radius
L = 15                   # Tube length

Exercise = 1

# C1: Follow a particle trajectory to a wall of the tube and then give it a new random trajectory. Follow the particle until it exits either 
# end of the tube. Provide the code you used as proof of your work. Provide a visual representation of 25 random paths.
if Exercise == 1:
    Nplot = 25               # Number of particles whose trajectories should be plotted

    entryRad = R*np.random.rand(Nplot)         # A vector of the initial radial distances of the particles from the centre of the tube when they enter it
    entryAngle = 2*np.pi*np.random.rand(Nplot) # A vector of the initial angles that describe the points where the particles enter the tube (angle as a polar 
                                           # coordinate of position, not angle as a component of initial trajectory), with 0 pointing upward
    trajAz = 2*np.pi*np.random.rand(Nplot)     # A vector of the azimuthal angles of the particles' initial trajectories, with 0 being aligned with the radius
    trajAlt = np.pi*np.random.rand(Nplot) - np.pi/2   # A vector of the altitude angles of the particles' initial trajectories, with 0 being aligned with the pipe axis. For this 
                                                  # exercise, we only care about angles that enter, so the angles are strictly between -pi/2 and pi/2.

    connectDots = np.empty((Nplot,3,1)) # This matrix will store all the coordinates of the points where particles hit the walls, for plotted trajectories
    
    for plotted in range(Nplot):
        z = np.abs(np.sqrt(R**2 + (entryRad[plotted])**2 - 2*R*entryRad[plotted]*np.cos(trajAz[plotted]))/np.tan(trajAlt[plotted])) # This is the axial distance the particle travels 
        # before its first collision with the wall.
        dist = np.sqrt(R**2 + (entryRad[plotted])**2 - 2*R*entryRad[plotted]*np.cos(trajAz[plotted]))/np.sin(trajAlt[plotted]) # This is the total distance it travels before the 
        # first collision.
        if z > L:
            dist = dist*L/z # If the particle shoots through the whole pipe during the initial step, rescale the distance so only the distance travelled through the pipe is counted
        
        connectDots[plotted,0,0] = entryRad[plotted] # Saves the initial r-coordinate of the plotted trajectory
        connectDots[plotted,1,0] = entryAngle[plotted] # Saves the initial theta-coordinate of the plotted trajectory
        j = 0 # Keeps track of how many times the while loop below has run
        theta = trajAz[plotted] # Angular coordinate of the particle when it makes its first collision with the wall
        phi = trajAlt[plotted]

        while z > 0: # Iterates until the particle leaves the tube from the direction it came
            j = j + 1
            prevDist = dist # Stores the total distance the particle had travelled prior to this step
            prevZ = z # Stores the total axial distance the particle had travelled prior to this step
            if connectDots.shape[2] <= j:
                connectDots = np.concatenate((connectDots, connectDots[:,:,-1][:,:,np.newaxis]),axis=2) # Extends the connectDots tensor if it's not large enough to store all the points, by copying the 
                                                                # previous set of points to the end
            connectDots[plotted,0,j] = R # The particle's radial position will always equal the pipe radius when it hits a wall
            connectDots[plotted,1,j] = theta # The particle's angular position
            connectDots[plotted,2,j] = z # The particle's axial position
            if z > L: # This means the particle exited out the back of the tube
                break

            phi = np.pi*np.random.rand(1) # Picks a random angle wrt to the axis to reflect of off the wall
            alpha = np.pi*np.random.rand(1) # Picks a random angle wrt to the tangent line perpendicular to the axis to reflect off the wall
            theta = theta - 2*alpha # This will be the new angular coordinate when the particle hits the wall next
            z += np.abs(2*R*np.sin(alpha))/np.tan(phi) # This is the additional axial distance the particle will travel
            if z > L:
                dist += ((L-prevZ)/z)*np.abs(2*R*np.sin(alpha)/np.sin(phi)) # Only add the distance the particle travelled while inside the tube if it makes it out
            else:
                dist += np.abs(2*R*np.sin(alpha)/np.sin(phi)) # This is the additional total distance the particle will travel

        # Plots the point after the particle escapes out the front of the tube
        if connectDots.shape[2] <= (j+1):
            connectDots = np.concatenate((connectDots, connectDots[:,:,-1][:,:,np.newaxis]), axis=2) # Extends the connectDots tensor if it's not large enough to store all the points, by copying the 
                                                            # previous set of points to the end
        connectDots[plotted,0,j+1] = R # The particle's radial position will always equal the pipe radius when it hits a wall
        connectDots[plotted,1,j+1] = theta # The particle's angular position
        connectDots[plotted,2,j+1] = z # The particle's axial position

        connectDots[plotted,2,0] = 0 # Saves the initial axial distance of the plotted trajectory, 0
        plotted = plotted + 1 # Moves onto the next path to plot       

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for k in range(Nplot):                      # Plot the particle paths
        a = connectDots.shape[2]
        x = connectDots[k,2,:]
        y = connectDots[k,0,:]*np.cos(connectDots[k,1,:])
        z = -connectDots[k,0,:]*np.sin(connectDots[k,1,:])

        for h in range(a-1):                    # This stops the script from plotting values of 0 after the particle escapes, making it appear to return to the tube
            if np.abs(x[h+1]) < 0.00001:        # Notice this doesn't check if the very first coordinate is zero, since we want this to be the case for axial distance at least
                x[h+1] = x[h]
            if np.abs(y[h+1]) < 0.00001:
                y[h+1] = y[h]            
            if np.abs(z[h+1]) < 0.00001:
                z[h+1] = z[h] 
        ax.plot3D(x,y,z)
        ax.set(xlim=(-5,20))

    X = np.arange(0,16,3)                       # Plot the cylindrical tube
    Y = np.arange(-R,2.6,0.5)
    X, Y = np.meshgrid(X,Y)
    Z1 = np.sqrt(R**2 - Y**2)                   # Upper half cylinder
    Z2 = -np.sqrt(R**2 - Y**2)                  # Lower half cylinder

    ax.plot_wireframe(X,Y,Z1,color='dimgrey') 
    ax.set(xlim=(-5,20))
    ax.plot_wireframe(X,Y,Z2,color='dimgrey') 
    ax.set(xlim=(-5,20))
    plt.show()


# C2: Estimate the fraction of particles that should exit the right end on average. You will need to simulate a large number of particle trajectories to determine this fraction
# with high precision. Try 25 particles, 100 particles, 250 particles, 1,000 particles… until you converge on the solution. Show a plot of this convergence. (hint: approximately 
# ¼ should exit, from the vacuum text above).
if Exercise == 2:
    fig = plt.figure()
    plt.title('Fraction of Particles That Make It Through the Tube')
    plt.xlabel('Particles Simulated')
    plt.ylabel('Success Percentage')
    ax = plt.axes()
    numTrials = 10
    N = np.array([25, 100, 250, 1000, 10000, 100000]) # Stores the number of particles to simulate each trial

    for u in range(numTrials):
        successRate = np.zeros(len(N)) # Stores the proportion of particles that escape out the far side for each trial

        for o in range(len(N)):
            success = np.zeros(N[o])    # A vector that keeps track of which particles make it out of the other side of the tube and which don't

            entryRad = R*np.random.rand(N[o])         # A vector of the initial radial distances of the particles from the centre of the tube when they enter it
            entryAngle = 2*np.pi*np.random.rand(N[o]) # A vector of the initial angles that describe the points where the particles enter the tube (angle as a polar 
                                                # coordinate of position, not angle as a component of initial trajectory), with 0 pointing upward
            trajAz = 2*np.pi*np.random.rand(N[o])     # A vector of the azimuthal angles of the particles' initial trajectories, with 0 being aligned with the radius
            trajAlt = 2*np.pi*np.random.rand(N[o])    # A vector of the altitude angles of the particles' initial trajectories, with 0 being aligned with the pipe axis

            for i in range(N[o]):
                if np.cos(trajAlt[i]) <= 0:
                    i = i+1                 # If the cosine of the altitude angle is negative, the angle is between pi/2 and 3pi/2 and the particle is pointing away from
                                            # the entrance to the pipe. It will therefore never enter and the particle already counts as a failure, so the script moves onto
                                            # the next particle
                else:
                    z = np.abs(np.sqrt(R**2 + (entryRad[i])**2 - 2*R*entryRad[i]*np.cos(trajAz[i]))/np.tan(trajAlt[i])) # This is the axial distance the particle travels before its first
                                                                                                            # collision with the wall.
                    dist = np.sqrt(R**2 + (entryRad[i])**2 - 2*R*entryRad[i]*np.cos(trajAz[i]))/np.sin(trajAlt[i]) # This is the total distance it travels before the first collision.
                    if z > L:
                        dist = dist*L/z # If the particle shoots through the whole pipe during the initial step, rescale the distance so only the distance travelled through the pipe is counted
                    
                    theta = trajAz[i] # Angular coordinate of the particle when it makes its first collision with the wall
                    phi = trajAlt[i]

                    while z > 0: # Iterates until the particle leaves the tube from the direction it came
                        prevDist = dist # Stores the total distance the particle had travelled prior to this step
                        prevZ = z # Stores the total axial distance the particle had travelled prior to this step
                        if z > L: # This means the particle exited out the back of the tube
                            success[i] = 1 # Indicates that the particle exited successfully
                            break

                        phi = np.pi*np.random.rand(1) # Picks a random angle wrt to the axis to reflect of off the wall
                        alpha = np.pi*np.random.rand(1) # Picks a random angle wrt to the tangent line perpendicular to the axis to reflect off the wall
                        theta = theta - 2*alpha # This will be the new angular coordinate when the particle hits the wall next
                        z += np.abs(2*R*np.sin(alpha))/np.tan(phi) # This is the additional axial distance the particle will travel
                        if z > L:
                            dist += ((L-prevZ)/z)*np.abs(2*R*np.sin(alpha)/np.sin(phi)) # Only add the distance the particle travelled while inside the tube if it makes it out
                        else:
                            dist += np.abs(2*R*np.sin(alpha)/np.sin(phi)) # This is the additional total distance the particle will travel            
            
                i = i + 1
            
            successRate[o] = 100*np.sum(success)/N[o]

        ax.plot(N,successRate)
    
    lineX = np.arange(0,N[-1]+10,5)
    lineY = 25*np.ones(len(lineX))
    ax.plot(lineX,lineY,'-',color='black')
    plt.show()


# C3: Keep track of the distance traveled for each particle in its “15 cm random walk” for those that do get through the tube’s output end. Plot this as a histogram of number of 
# particles vs travel distance in units of 15cm. At a typical velocity expected for room temperature air molecules, determine the average speed for particles to traverse the tube.
if Exercise == 3:
    N = 10000 # Number of particles to simulate
    totDist = np.zeros(N) # Keeps track of the total distance each particle travels
    success = np.zeros(N)    # A vector that keeps track of which particles make it out of the other side of the tube and which don't

    entryRad = R*np.random.rand(N)         # A vector of the initial radial distances of the particles from the centre of the tube when they enter it
    entryAngle = 2*np.pi*np.random.rand(N) # A vector of the initial angles that describe the points where the particles enter the tube (angle as a polar 
                                        # coordinate of position, not angle as a component of initial trajectory), with 0 pointing upward
    trajAz = 2*np.pi*np.random.rand(N)     # A vector of the azimuthal angles of the particles' initial trajectories, with 0 being aligned with the radius
    trajAlt = 2*np.pi*np.random.rand(N)    # A vector of the altitude angles of the particles' initial trajectories, with 0 being aligned with the pipe axis

    for i in range(N):
        if np.cos(trajAlt[i]) <= 0:
            i = i+1                 # If the cosine of the altitude angle is negative, the angle is between pi/2 and 3pi/2 and the particle is pointing away from
                                    # the entrance to the pipe. It will therefore never enter and the particle already counts as a failure, so the script moves onto
                                    # the next particle
        else:
            z = np.abs(np.sqrt(R**2 + (entryRad[i])**2 - 2*R*entryRad[i]*np.cos(trajAz[i]))/np.tan(trajAlt[i])) # This is the axial distance the particle travels before its first
                                                                                                    # collision with the wall.
            dist = np.sqrt(R**2 + (entryRad[i])**2 - 2*R*entryRad[i]*np.cos(trajAz[i]))/np.sin(trajAlt[i]) # This is the total distance it travels before the first collision.
            if z > L:
                dist = dist*L/z # If the particle shoots through the whole pipe during the initial step, rescale the distance so only the distance travelled through the pipe is counted
            
            theta = trajAz[i] # Angular coordinate of the particle when it makes its first collision with the wall
            phi = trajAlt[i]

            while z > 0: # Iterates until the particle leaves the tube from the direction it came
                prevZ = z # Stores the total axial distance the particle had travelled prior to this step
                if z > L: # This means the particle exited out the back of the tube
                    success[i] = 1 # Indicates that the particle exited successfully
                    totDist[i] = dist # Updates the total distance of the particle
                    break

                phi = np.pi*np.random.rand(1) # Picks a random angle wrt to the axis to reflect of off the wall
                alpha = np.pi*np.random.rand(1) # Picks a random angle wrt to the tangent line perpendicular to the axis to reflect off the wall
                theta = theta - 2*alpha # This will be the new angular coordinate when the particle hits the wall next
                z += np.abs(2*R*np.sin(alpha))/np.tan(phi) # This is the additional axial distance the particle will travel
                if z > L:
                    dist += ((L-prevZ)/z)*np.abs(2*R*np.sin(alpha)/np.sin(phi)) # Only add the distance the particle travelled while inside the tube if it makes it out
                else:
                    dist += np.abs(2*R*np.sin(alpha)/np.sin(phi)) # This is the additional total distance the particle will travel            
    
        i = i + 1

    histDist = success*totDist # This will cancel out any distances for particles that escaped out the front that were counted anyway
    histDist = histDist[histDist >= 15] # The total distance must be greater than or equal to the length of the tube
    
    numBins = int(np.max(histDist) // 15) # Split into bins 15 cm wide

    plt.hist(histDist, bins=numBins, range=(14.999, 15*(numBins+1)))
    plt.title('Number of Particles vs Travel Distance (cm)')
    plt.show()

    print('The average travel distance is: ' + str(np.mean(histDist)) + ' cm')


# C4: Keep track of the final velocity that the particles leave the exit end. Divide these into 10 degree increments of the trajectory, relative to the tube axis and plot the 
# distribution. What is the approximate angular spread of the molecular beam produced.
if Exercise == 4:
    N = 10000 # Number of particles to simulate
    finAngle = np.full(N,np.nan) # Keeps track of the final angle of the escaping particles. Intentionally initialized to NaN, and only the ones that escape will have it updated to
    # a real number
    success = np.zeros(N)    # A vector that keeps track of which particles make it out of the other side of the tube and which don't

    entryRad = R*np.random.rand(N)         # A vector of the initial radial distances of the particles from the centre of the tube when they enter it
    entryAngle = 2*np.pi*np.random.rand(N) # A vector of the initial angles that describe the points where the particles enter the tube (angle as a polar 
                                        # coordinate of position, not angle as a component of initial trajectory), with 0 pointing upward
    trajAz = 2*np.pi*np.random.rand(N)     # A vector of the azimuthal angles of the particles' initial trajectories, with 0 being aligned with the radius
    trajAlt = 2*np.pi*np.random.rand(N)    # A vector of the altitude angles of the particles' initial trajectories, with 0 being aligned with the pipe axis

    for i in range(N):
        if np.cos(trajAlt[i]) <= 0:
            i = i+1                 # If the cosine of the altitude angle is negative, the angle is between pi/2 and 3pi/2 and the particle is pointing away from
                                    # the entrance to the pipe. It will therefore never enter and the particle already counts as a failure, so the script moves onto
                                    # the next particle
        else:
            z = np.abs(np.sqrt(R**2 + (entryRad[i])**2 - 2*R*entryRad[i]*np.cos(trajAz[i]))/np.tan(trajAlt[i])) # This is the axial distance the particle travels before its first
                                                                                                    # collision with the wall.
            dist = np.sqrt(R**2 + (entryRad[i])**2 - 2*R*entryRad[i]*np.cos(trajAz[i]))/np.sin(trajAlt[i]) # This is the total distance it travels before the first collision.
            if z > L:
                dist = dist*L/z # If the particle shoots through the whole pipe during the initial step, rescale the distance so only the distance travelled through the pipe is counted
            
            theta = trajAz[i] # Angular coordinate of the particle when it makes its first collision with the wall
            phi = trajAlt[i]

            while z > 0: # Iterates until the particle leaves the tube from the direction it came
                prevZ = z # Stores the total axial distance the particle had travelled prior to this step
                if z > L: # This means the particle exited out the back of the tube
                    success[i] = 1 # Indicates that the particle exited successfully
                    finAngle[i] = theta # Updates the total distance of the particle
                    break

                phi = np.pi*np.random.rand(1) # Picks a random angle wrt to the axis to reflect of off the wall
                alpha = np.pi*np.random.rand(1) # Picks a random angle wrt to the tangent line perpendicular to the axis to reflect off the wall
                theta = theta - 2*alpha # This will be the new angular coordinate when the particle hits the wall next
                z += np.abs(2*R*np.sin(alpha))/np.tan(phi) # This is the additional axial distance the particle will travel
                if z > L:
                    dist += ((L-prevZ)/z)*np.abs(2*R*np.sin(alpha)/np.sin(phi)) # Only add the distance the particle travelled while inside the tube if it makes it out
                else:
                    dist += np.abs(2*R*np.sin(alpha)/np.sin(phi)) # This is the additional total distance the particle will travel            
    
        i = i + 1

    histAngle = finAngle[np.isnan(finAngle) == 0] # Only keeps values that aren't inf, ie, those that have a valid angle
    histAngle = histAngle % (2*np.pi) # Reduces all angles mod 2pi so they all fall within the range [0,2pi)
    histAngle = np.degrees(histAngle) # Converts to degrees

    plt.hist(histAngle, bins=36)
    plt.xticks(np.arange(0,370,10))
    plt.title('Number of Particles vs Angle of Escape (degrees)')
    plt.show()

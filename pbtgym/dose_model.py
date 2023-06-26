from .requirements import *

def bit_wise_dose(self) -> float:
    """
    Applies dose (currently just a fixed amount) at given angle and position.
    Currently just going to place 3x3 dose centered on beam position in order
    to implement proof of concept. Then can increase levels of complexity
    from there.
    """
    square_dose = self.default_dose * np.ones((3, 3))
    x = self.beam.x
    y = self.beam.y
    dose_overlap = np.sum(self.reward_map[(y-1):(y+2),(x-1):(x+2)] * square_dose)
    score = dose_overlap / (9 * self.default_dose)
    self.canvas[(y-1):(y+2),(x-1):(x+2)] += square_dose
    return score


def complex_dose(self) -> float:
    """
    Creates a line of a given thickness between two points and applies a 
    custom gradient to said line (function of distance along the line).
    Then calculates a normalised reward based on the overlap of dose and
    contour.
    """
    x = self.beam.x
    y = self.beam.y
    theta = self.beam.theta
    # Defining properties of each circle to be drawn
    radius = 2
    n=0
    score = 0
    if not self.within_patient(x,y):# Just double check this later
    #     print("Not within patient")
        score -= 0.1
    # else:
    #     print("Within target")

    if theta == 0:
        while(self.within_patient(x+n,y)):
            value = self.beam.calculate_dose_gradient(n)
            self.canvas[y+n,x-radius:x+radius+1] += value
            score += np.sum(self.reward_map[y+n,x-radius:x+radius+1]) * value / (2*radius + 1)
            n += 1

    elif theta == 180:
        while(self.within_patient(x-n,y)):
            value = self.beam.calculate_dose_gradient(n)
            self.canvas[y-n,x-radius:x+radius+1] += value
            score += np.sum(self.reward_map[y-n,x-radius:x+radius+1]) * value / (2*radius + 1)
            n += 1

    elif theta == 90:
        while(self.within_patient(x,y+n)):
            value = self.beam.calculate_dose_gradient(n)
            self.canvas[y-radius:y+radius+1,x+n] += value
            score += np.sum(self.reward_map[y-radius:y+radius+1,y+n]) * value / (2*radius + 1) # Multiplicative term at the end here is a normalisation factor (value [0,1], 2x Radius + 1 pixels per step)
            n += 1

    elif theta == 270:
        while(self.within_patient(x,y-n)):
            value = self.beam.calculate_dose_gradient(n)
            self.canvas[y-radius:y+radius+1,y-n] += value
            score += np.sum(self.reward_map[y-radius:y+radius+1,x-n]) * value / (2*radius + 1)
            n += 1

    elif theta < 90 or theta > 270:
        m = np.rint(np.tan(theta*np.pi/180))
        while(self.within_patient(x+n,int(y+m*n))):
            distance = n * np.sqrt(1 + m**2) # should always be sqrt(2) x n for this model but writing like this in case theta precision changes
            value = self.beam.calculate_dose_gradient(distance)
            for r in range(2*radius+1):
                k = r - radius
                self.canvas[y+n-k, int(x+m*(n+k))] += value
                score += np.sum(self.reward_map[y+n-k, int(x+m*(n+k))]) * value / (2*radius + 1)  
            for r in range(2*radius):
                k = r - radius
                self.canvas[y+n+k, int(x+m*(n-k-1))] += value
                score += np.sum(self.reward_map[y+n+k, int(x+m*(n-k-1))]) * value / (2*radius)
            n += 1

    elif theta > 90 and theta < 270:
        m = np.rint(np.tan(theta*np.pi/180))
        while(self.within_patient(x-n,int(y-m*n))):
            distance = n * np.sqrt(1 + m**2) # should always be sqrt(2) x n for this model but writing like this in case theta precision changes
            value = self.beam.calculate_dose_gradient(distance)
            for r in range(2*radius+1):
                k = r - radius
                self.canvas[y-n-k,int(x-m*(n-k))] += value
                score += np.sum(self.reward_map[y-radius:y+radius+1,x-n]) * value / (2*radius + 1)

            for r in range(2*radius):
                k = r - radius
                self.canvas[y-n+k, int(x-m*(n+k+1))] += value
                score += np.sum(self.reward_map[y-n+k, int(x-m*(n+k+1))]) * value / (2*radius)
            n += 1
    return score

def apply_bortfeld_dose(self):
    """
    Creates a line of a given thickness between two points and applies a 
    custom gradient to said line (function of distance along the line).
    Then calculates a normalised reward based on the overlap of dose and
    contour.
    """
    x = self.beam.x
    y = self.beam.y
    theta = self.beam.theta        
    
    radius = 2
    n=0
    increments = np.array([])

    if theta == 0:
        # Move in a straight line in y-direction
        if self.within_patient(y+n, x):
            # Move until the beam exits the patient (apprximating 0 attentuation outside the patient)
            while(self.within_patient(y+n, x)):
                increments = np.append(increments, n)
                n += 1
            # Record distance (depth) into patient
            depth = np.copy(n)
            n = 0
            # Calculate values at each point along the line using the Bortfeld approximation
            values = self.calculate_bortfield_gradient(depth - increments, increments[-1])
            # Increment the canvas with each value along the line
            for value in values:
                self.canvas[y+n,x-radius:x+radius+1] += value
                n += 1

    elif theta == 180 & self.within_patient(y-n,x):
        while(self.within_patient(y-n,x)):
            increments = np.append(increments, n)
            n += 1
        depth = np.copy(n)
        n = 0
        values = self.calculate_bortfield_gradient(depth - increments, depth)
        for value in values:
            self.canvas[y-n,x-radius:x+radius+1] += value
            n += 1

    elif theta == 90 & self.within_patient(y,x+n):
        while(self.within_patient(y,x+n)):
            increments = np.append(increments, n)
            n += 1
        depth = np.copy(n)
        n = 0
        values = self.calculate_bortfield_gradient(increments, depth)
        for value in values:
            self.canvas[y-radius:y+radius+1,x+n] += value
            n += 1

    elif theta == 270 & self.within_patient(y,x-n):
        while(self.within_patient(y,x-n)):
            increments = np.append(increments, n)
            n += 1
        depth = np.copy(n)
        n = 0
        values = self.calculate_bortfield_gradient(depth - increments, depth)
        for value in values:
            self.canvas[y-radius:y+radius+1,x-n] += value
            n += 1

    elif theta < 90 or theta > 270:
        m = np.rint(np.tan(theta*np.pi/180))
        if self.within_patient(y+n,int(x+m*n)):
            while(self.within_patient(y+n,int(x+m*n))):
                distance = n * np.sqrt(1 + m**2)
                increments = np.append(increments, distance)
                n += 1
            depth = np.copy(distance)
            n = 0
            values = self.calculate_bortfield_gradient(depth - increments, depth)
            for value in values:
                for r in range(2*radius+1):
                    k = r - radius
                    self.canvas[y+n-k, int(x+m*(n+k))] += value
                for r in range(2*radius):
                    k = r - radius
                    self.canvas[y+n+k, int(x+m*(n-k-1))] += value
                n += 1

    elif theta > 90 and theta < 270:
        m = np.rint(np.tan(theta*np.pi/180))
        if self.within_patient(y-n,int(x-m*n)):
            while(self.within_patient(y-n,int(x-m*n))):
                distance = n * np.sqrt(1 + m**2)
                increments = np.append(increments, distance)
                n += 1
            depth = np.copy(distance)
            n = 0
            values = self.calculate_bortfield_gradient(depth - increments, depth)
            for value in values:
                for r in range(2*radius+1):
                    k = r - radius
                    self.canvas[y-n-k,int(x-m*(n-k))] += value
                for r in range(2*radius):
                    k = r - radius
                    self.canvas[y-n+k, int(x-m*(n+k+1))] += value
                n += 1

    # Returns zero as reward function is now state-evaluation dependent, potential for change in future
    return 0

def calculate_bortfield_gradient(self, distance, range):
    sigma = 0.27    #range straggling sigma
    epsilon = 0.2   #low energy contamination
    p = 1.77    #exponent of range-energy relationship
    if range == 0:
        return [0]
    out = 0.05*0.65*(self.cyl_gauss(-1/p,(distance-range)/sigma)+sigma*(0.01394+epsilon/range)*self.cyl_gauss(-1/p-1,(distance-range)/sigma))
    
    return out

def cyl_gauss(self, a, x):
    "Calculate product of Gaussian and parabolic cylinder function"
    y = np.copy(x)
    branch = -12.0   #for large negative values of the argument we run into numerical problems, need to approximate result
    x1 = x[np.where(x<branch)]
    y1 = math.sqrt(2*math.pi)/special.gamma(-a)*(-x1)**(-a-1)
    y[np.where(x<branch)] = y1

    x2 = x[np.where(x>=branch)]
    y2a = special.pbdv(a,x2)[0]     #special function yielding parabolic cylinder function, first array [0] is function itself
    y2b = np.exp(-x2*x2/4)
    y2 = y2a*y2b

    y[np.where(x>=branch)] = y2

    return y
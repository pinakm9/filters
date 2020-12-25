import numpy as np
from numpy.linalg import multi_dot,inv
from scipy.integrate import solve_ivp  # to be replaced by solve_ivp later

def fun(t,state,args_):
    "Function to be used for compuation of ode in scipy.integrate.solve_ivp"
    x,y,z=state
    sigma,rho,beta=args_
    return sigma*(y-x),x*(rho-z)-y,x*y-beta*z

class lorenz_63:
    """create a class to solve lorenz 63 system.create objects with attributes
    variables-number of variables  ;sigma,rho,beta: the parameters ;
    time_start= starting time of solution ; time_stop=ending time of the solution initial
    ,the initial value of the system at t start."""

    def __init__(self,parameters):
        "To intialize the variables "
        # parse parameters from the dictionary
        for key in parameters:
            setattr(self, key, parameters[key])
        #self.time_units=int((self.time_stop-self.time_start)/self.time_step)
        #self.t_evaluate=None

    def solution(self):
        "To generate the full solution from starting time to stop time"
        flow=solve_ivp(fun,[self.time_start,self.time_stop],self.initial,method='RK45',t_eval=self.t_evaluate,args=([self.sigma,self.rho,self.beta],))
        self.soln=(flow.y).T
        #self.time=flow.t
        return self.soln


        from lorenz_63_ode import *
        #from II_scale_lorenz_96 import *
        import numpy as np
        import json
        seed_num=45
        np.random.seed(seed_num)

        #Now,load the initial condition,and integrate the ode:

        """Loading parameters and initial condition for lorenz_63"""
        parametersFile='L_63_x0_seed_{}.json'.format(seed_num)
        with open(parametersFile) as jsonfile:
             parameters=json.load(jsonfile)

        parameters['time_start']=0
        parameters['time_stop']=100

        #parameters['obs_gap']=0.1
        #parameters['t_evaluate']=np.arange(0,100,0.1)

        """Generate trajectory for various time resolutions"""
        obs_gaps_=np.array([0.2])
        os.chdir(r'home/shashank/Lorenz_63/Trajectory_Observations')
        #integrate to generate the trajectory:
        for i in obs_gaps_:
            parameters['obs_gap']=i
            parameters['t_evaluate']=np.arange(0,parameters['time_stop'],i)
            ob1=lorenz_63(parameters)
            np.save('Trajectory_{}_.npy'.format(parameters['obs_gap']),ob1.solution())

        print('Job Done')

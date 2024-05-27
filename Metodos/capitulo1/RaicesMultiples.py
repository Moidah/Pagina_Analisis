import numpy as np 
import sympy as sym

def RaicesMul(x0,tol,n,f,derivada1,derivdad2):
    fx0 = f(x0)
    if fx0 == 0:
        return x0
    else:
        fx0 = f(x0)
        fpx0 = derivada1(x0)
        fppx0 = derivdad2(x0)
        i = 0 
        error = tol +1 
        den = (fpx0**2)-(fx0*fppx0)
        while error>tol and fx0 != 0 and den != 0 and i <n:
            x1 = x0 - ((fx0*fpx0)/den)           
            fx0 = f(x1)
            fpx0 = derivada1(x1)
            fppx0 = derivdad2(x1)
            den = (fpx0**2)-(fx0*fppx0)
            error = np.abs(x1-x0)
            x0 = x1
            i += 1 
        if fx0 == 0:
            return x0 
        elif error<tol:
            return [x0,error]
        else:
            return  [x0,"Fracasó en "+str(n)+" iteraciones"]


import numpy as np 

def secante(x0,x1,tol,n,f):
    fx0 = f(x0)
    if fx0 == 0:
        return x0
    else:
        fx1 = f(x1)
        i = 0 
        error = tol +1 
        den = fx1-fx0
        while error>tol and fx1 != 0 and den != 0 and i <n:
            x2 = x1 - ((fx1*(x1-x0))/den)
            error = np.abs(x2-x1)
            x0 = x1
            fx0 = fx1
            x1= x2
            fx1 = f(x1)
            den = fx1-fx0
            i += 1 
        if fx1 == 0:
            return [x1,"Eror de "+str(0)]
        elif error<tol:
            return [x1,"Eror de "+str(error)]
        elif den == 0:
            return ["Posible raíz multiple"]
        else:
            return  [x1,"Fracasó en "+str(n)+" iteraciones"]


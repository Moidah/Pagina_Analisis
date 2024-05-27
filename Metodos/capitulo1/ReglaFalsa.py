import numpy as np

def ReglaF(xi,xs,tol,n,f):
    fxi = f(xi)
    fxs = f(xs)
    if fxi == 0: 
        return xi 
    elif fxs == 0:
        return xs
    elif fxi*fxs < 0: 
        xm = xi-((fxi*(xs-xi)))/(fxs-fxi)
        fxm = f(xm)
        i = 1 
        error = tol +1 
        while error > tol and fxm != 0 and i < n:
            if fxi* fxm < 0:
                xs = xm
                fxs = fxm
            else:
                xi = xm
                fxi = fxm
            xaux = xm 
            xm = xi-((fxi*(xs-xi)))/(fxs-fxi)
            fxm = f(xm)
            error = np.abs(xm-xaux)
            i += 1
        if fxm == 0:
            return [xm,"Error de "+str(0)]
        elif error<tol:
            return [xm,"Error de "+str(error)]
        else:
            return ["Fracasó en "+str(n)+" iteraciones"]
    else:
        return ["Intervalo inadecaudo"]
    
    
ReglaF(5.0,15.0,1e-6,100,x^2-100)
    

def sor_matricial(A,b,x0,n,tol,w):
    table=PrettyTable()
    table.field_names=["Iteraciones","Vector","Tolerancia"]
    it=0
    t=tol+1
    table.add_rows([[it,x0,t]])
    D=np.diag(np.diag(A))
    L=(-1)*np.tril(A-D)
    U=(-1)*np.triu(A-D)
    T=np.linalg.inv(D-w*L)@((1-w)*D+w*U)
    C=w*np.linalg.inv(D-w*L)@np.transpose(b)
    while t>tol:
        it+=1
        x_nuevo=np.transpose(T@np.transpose(x0)+C)
        t=max(abs(x0-x_nuevo))
        x0=x_nuevo
        table.add_rows([[it,x0,t]])
    print(table)
    
sor_matricial(np.array([[45,13,-4,8],[-5,-28,4,-14],[9,15,63,-7],[2,3,-8,-42]]),np.array([-25,82,75,-43]),np.array([2,2,2,2]),4,1e-5,1)
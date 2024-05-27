def gauss_seidel(A,b,x0,n,tol):
    table=PrettyTable()
    table.field_names=["Iteraciones","Vector","Tolerancia"]
    it=0
    t=tol+1
    table.add_rows([[it,x0,t]])
    while t>tol:
        it+=1
        x_nuevo=np.zeros(n)
        for j in range(n):
            for k in range(n):
                if(j==k):
                    continue
                if(j>k):
                    x_nuevo[j]+=(-1)*A[j][k]*x_nuevo[k]
                else:
                    x_nuevo[j]+=(-1)*A[j][k]*x0[k]
            x_nuevo[j]+=b[j]
            x_nuevo[j]/=A[j][j]
        t=max(abs(x0-x_nuevo))
        x0=x_nuevo
        table.add_rows([[it,x0,t]])
    print(table)
gauss_seidel(np.array([[45,13,-4,8],[-5,-28,4,-14],[9,15,63,-7],[2,3,-8,-42]]),np.array([-25,82,75,-43]),np.array([2,2,2,2]),4,1e-5)
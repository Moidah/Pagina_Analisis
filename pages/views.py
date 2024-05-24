from django.shortcuts import render, HttpResponse
from django import forms
from django.views.generic import TemplateView
import plotly.graph_objs as go
from plotly.offline import plot
import numpy as np
from numpy.linalg import inv
import sympy as sp

class FunctionForm(forms.Form):
    function = forms.CharField(label='Función', widget=forms.TextInput(attrs={'placeholder': 'Ingrese la función en términos de x'}))

class HomePageView(TemplateView):
    template_name = 'home.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = FunctionForm()
        return context

    def post(self, request, *args, **kwargs):
        form = FunctionForm(request.POST)
        if form.is_valid():
            function_str = form.cleaned_data['function']
            x = sp.symbols('x')
            try:
                # Convertir la cadena de texto a una expresión simbólica
                function_expr = sp.sympify(function_str)
                # Generar valores de x
                x_values = np.linspace(-10, 10, 400)
                # Evaluar la función para cada valor de x
                y_values = [function_expr.subs(x, val) for val in x_values]

                # Crear gráfico
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name='Función'))

                plot_div = plot(fig, output_type='div', include_plotlyjs=False)
                context = self.get_context_data()
                context['form'] = form
                context['plot_div'] = plot_div
                return render(request, self.template_name, context)
            except (sp.SympifyError, TypeError):
                form.add_error('function', 'Ingrese una función válida en términos de x.')
                context = self.get_context_data()
                context['form'] = form
                return render(request, self.template_name, context)
        else:
            context = self.get_context_data()
            context['form'] = form
            return render(request, self.template_name, context)

class MetodosPageView(TemplateView):
    template_name = 'metodos.html'

class NosotrosPageView(TemplateView):
    template_name = 'nosotros.html'


#-------------------------------------METODOS DESDE AQUI-------------------------------------------------------------------------
#----------BISECCION----------------------
class BiseccionForm(forms.Form):
    xi = forms.FloatField(label='Xi', required=True)
    xs = forms.FloatField(label='Xs', required=True)
    tol = forms.FloatField(label='Tol', required=True)
    niter = forms.IntegerField(label='Niter', required=True)
    fun = forms.CharField(label='Function', required=True, widget=forms.TextInput(attrs={'placeholder': 'Ingrese la función en términos de x'}))
    export = forms.BooleanField(label='Exportar resultados a TXT', required=False)
    
class BiseccionPageView(TemplateView):
    template_name = 'biseccion.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = BiseccionForm()
        return context

    def post(self, request, *args, **kwargs):
        form = BiseccionForm(request.POST)
        context = self.get_context_data()
        if form.is_valid():
            xi = form.cleaned_data['xi']
            xs = form.cleaned_data['xs']
            tol = form.cleaned_data['tol']
            niter = form.cleaned_data['niter']
            fun = form.cleaned_data['fun']
            export = form.cleaned_data['export']

            x = sp.symbols('x')
            fi = sp.sympify(fun).subs(x, xi)
            fs = sp.sympify(fun).subs(x, xs)
            iterations = []

            if fi == 0:
                s = xi
                E = 0
                result = f"{xi} es raíz de f(x)"
            elif fs == 0:
                s = xs
                E = 0
                result = f"{xs} es raíz de f(x)"
            elif fs * fi < 0:
                c = 0
                Xm = (xi + xs) / 2
                fe = sp.sympify(fun).subs(x, Xm)
                fm = [fe]
                E = [100]

                while E[c] > tol and fe != 0 and c < niter:
                    row = [c + 1, xi, xs, Xm, fe, E[c]]
                    iterations.append(row)

                    if fi * fe < 0:
                        xs = Xm
                        fs = sp.sympify(fun).subs(x, xs)
                    else:
                        xi = Xm
                        fi = sp.sympify(fun).subs(x, xi)

                    Xa = Xm
                    Xm = (xi + xs) / 2
                    fe = sp.sympify(fun).subs(x, Xm)
                    fm.append(fe)
                    Error = abs(Xm - Xa)
                    E.append(Error)
                    c += 1

                row = [c + 1, xi, xs, Xm, fe, Error]
                iterations.append(row)

                if fe == 0:
                    s = Xm
                    result = f"{Xm} es raíz de f(x)"
                elif Error < tol:
                    s = Xm
                    result = f"{Xm} es una aproximación de una raíz de f(x) con una tolerancia {tol}"
                else:
                    result = f"Fracaso en {niter} iteraciones"
            else:
                result = "El intervalo es inadecuado"

            if export:
                response = HttpResponse(content_type='text/plain')
                response['Content-Disposition'] = 'attachment; filename="resultados_biseccion.txt"'
                response.write(f"Resultado: {result}\n")
                response.write("Iteración\tXi\tXs\tXm\tf(Xm)\tError\n")
                for row in iterations:
                    response.write("\t".join(map(str, row)) + "\n")
                return response

            context['result'] = result
            context['iterations'] = iterations
            context['form'] = form
        else:
            context['form'] = form

        return render(request, self.template_name, context)
    
#---------PUNTO_FIJO--------------------
class PuntoFijoForm(forms.Form):
    x0 = forms.FloatField(label='X0', required=True)
    tol = forms.FloatField(label='Tol', required=True)
    Nmax = forms.IntegerField(label='Nmax', required=True)
    g = forms.CharField(label='Function g(x)', required=True, widget=forms.TextInput(attrs={'placeholder': 'Ingrese la función g(x)'}))
    export = forms.BooleanField(label='Exportar resultados a TXT', required=False)
    
class PuntoFijoPageView(TemplateView):
    template_name = 'puntofijo.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = PuntoFijoForm()
        return context

    def post(self, request, *args, **kwargs):
        form = PuntoFijoForm(request.POST)
        context = self.get_context_data()
        if form.is_valid():
            x0 = form.cleaned_data['x0']
            tol = form.cleaned_data['tol']
            Nmax = form.cleaned_data['Nmax']
            g_str = form.cleaned_data['g']
            export = form.cleaned_data['export']

            x = sp.symbols('x')
            g = sp.sympify(g_str)
            f = (x)**2 - 100

            def Cre_o_Decre(f, x0):
                return sp.diff(f, x).subs(x, x0) > 0

            def PuntoFijo(g, x0, tol, Nmax):
                # Inicialización
                xant = x0
                E = 1000
                cont = 0
                iteraciones = []
                
                # Ciclo
                while E > tol and cont < Nmax:
                    xact = g.subs(x, xant)
                    E = abs(xact - xant)
                    cont += 1
                    iteraciones.append((cont, xant, xact, E))
                    xant = xact

                return [xact, cont, E], iteraciones

            creciente = Cre_o_Decre(f, x0)
            (xact, cont, E), iteraciones = PuntoFijo(g, x0, tol, Nmax)
            result = f"Convergió a {xact} con una tolerancia de {tol}" if E < tol else f"No convergió después de {Nmax} iteraciones"

            context['creciente'] = "La función es creciente en el punto inicial" if creciente else "La función es decreciente en el punto inicial"
            context['result'] = result
            context['iterations'] = iteraciones

            if export:
                response = HttpResponse(content_type='text/plain')
                response['Content-Disposition'] = 'attachment; filename="resultados_puntofijo.txt"'
                response.write(f"{context['creciente']}\n")
                response.write(f"Resultado: {result}\n")
                response.write("Iteración\tXant\tXact\tError\n")
                for row in iteraciones:
                    response.write("\t".join(map(str, row)) + "\n")
                return response

            context['form'] = form
        else:
            context['form'] = form

        return render(request, self.template_name, context)

#---------Regla_Falsa-------------------
class ReglaFalsaForm(forms.Form):
    xi = forms.FloatField(label='Xi', required=True)
    xs = forms.FloatField(label='Xs', required=True)
    tol = forms.FloatField(label='Tol', required=True)
    n = forms.IntegerField(label='Niter', required=True)
    fun = forms.CharField(label='Function', required=True, widget=forms.TextInput(attrs={'placeholder': 'Ingrese la función en términos de x'}))
    export = forms.BooleanField(label='Exportar resultados a TXT', required=False)
    

class ReglaFalsaPageView(TemplateView):
    template_name = 'reglafalsa.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = ReglaFalsaForm()
        return context

    def post(self, request, *args, **kwargs):
        form = ReglaFalsaForm(request.POST)
        context = self.get_context_data()
        if form.is_valid():
            xi = form.cleaned_data['xi']
            xs = form.cleaned_data['xs']
            tol = form.cleaned_data['tol']
            n = form.cleaned_data['n']
            fun_str = form.cleaned_data['fun']
            export = form.cleaned_data['export']

            x = sp.symbols('x')
            fun = sp.lambdify(x, sp.sympify(fun_str), 'numpy')

            def ReglaF(xi, xs, tol, n, f):
                fxi = f(xi)
                fxs = f(xs)
                table = []
                if fxi == 0:
                    return xi, table
                elif fxs == 0:
                    return xs, table
                elif fxi * fxs < 0:
                    xm = xi - ((fxi * (xs - xi))) / (fxs - fxi)
                    fxm = f(xm)
                    i = 1
                    error = tol + 1
                    while error > tol and fxm != 0 and i < n:
                        row = [i, xi, xs, xm, fxm, error]
                        table.append(row)
                        if fxi * fxm < 0:
                            xs = xm
                            fxs = fxm
                        else:
                            xi = xm
                            fxi = fxm
                        xaux = xm
                        xm = xi - ((fxi * (xs - xi)) / (fxs - fxi))
                        fxm = f(xm)
                        error = np.abs(xm - xaux)
                        i += 1
                    row = [i, xi, xs, xm, fxm, error]
                    table.append(row)
                    if fxm == 0:
                        return [xm, "Error de " + str(0)], table
                    elif error < tol:
                        return [xm, "Error de " + str(error)], table
                    else:
                        return ["Fracasó en " + str(n) + " iteraciones"], table
                else:
                    return ["Intervalo inadecuado"], table

            result, table = ReglaF(xi, xs, tol, n, fun)

            if export:
                response = HttpResponse(content_type='text/plain')
                response['Content-Disposition'] = 'attachment; filename="resultados_reglafalsa.txt"'
                response.write(f"Raíz aproximada: {result[0]}\n")
                response.write(f"{result[1]}\n")
                response.write("Iteración\tXi\tXs\tXm\tf(Xm)\tError\n")
                for row in table:
                    response.write("\t".join(map(str, row)) + "\n")
                return response

            context['result'] = result
            context['table'] = table
            context['form'] = form
        else:
            context['form'] = form

        return render(request, self.template_name, context)
    
#----------NEWTON-----------------------

class NewtonForm(forms.Form):
    x0 = forms.FloatField(label='X0', required=True)
    tol = forms.FloatField(label='Tol', required=True)
    Nmax = forms.IntegerField(label='Nmax', required=True)
    fun = forms.CharField(label='Function', required=True, widget=forms.TextInput(attrs={'placeholder': 'Ingrese la función en términos de x'}))
    export = forms.BooleanField(label='Exportar resultados a TXT', required=False)

class NewtonPageView(TemplateView):
    template_name = 'newton.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = NewtonForm()
        return context

    def post(self, request, *args, **kwargs):
        form = NewtonForm(request.POST)
        context = self.get_context_data()
        if form.is_valid():
            x0 = form.cleaned_data['x0']
            tol = form.cleaned_data['tol']
            Nmax = form.cleaned_data['Nmax']
            fun_str = form.cleaned_data['fun']
            export = form.cleaned_data['export']

            x = sp.symbols('x')
            f = sp.sympify(fun_str)

            def Newton(f, x0, tol, Nmax):
                xant = x0
                fant = f.subs(x, xant)
                E = 1000
                cont = 0
                iteraciones = []

                while E > tol and cont < Nmax:
                    xact = xant - fant / (sp.diff(f, x).subs(x, xant))
                    fact = f.subs(x, xant)
                    E = abs(xact - xant)
                    cont += 1
                    iteraciones.append((cont, float(xant), float(xact), float(fact), float(E)))
                    xant = xact
                    fant = fact

                return [float(xact), cont, float(E)], iteraciones

            result, iteraciones = Newton(f, x0, tol, Nmax)

            if export:
                response = HttpResponse(content_type='text/plain')
                response['Content-Disposition'] = 'attachment; filename="resultados_newton.txt"'
                response.write(f"Raíz aproximada: {result[0]}\n")
                response.write(f"Número de iteraciones: {result[1]}\n")
                response.write(f"Error: {result[2]}\n")
                response.write("Iteración\tXant\tXact\tf(Xact)\tError\n")
                for row in iteraciones:
                    response.write("\t".join(map(str, row)) + "\n")
                return response

            context['result'] = result
            context['iterations'] = iteraciones
            context['form'] = form
        else:
            context['form'] = form

        return render(request, self.template_name, context)
    
#---------SECANTE---------------------------------
class SecanteForm(forms.Form):
    x0 = forms.FloatField(label='X0', required=True)
    x1 = forms.FloatField(label='X1', required=True)
    tol = forms.FloatField(label='Tol', required=True)
    n = forms.IntegerField(label='Niter', required=True)
    fun = forms.CharField(label='Function', required=True, widget=forms.TextInput(attrs={'placeholder': 'Ingrese la función en términos de x'}))
    export = forms.BooleanField(label='Exportar resultados a TXT', required=False)
    
class SecantePageView(TemplateView):
    template_name = 'secante.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = SecanteForm()
        return context

    def post(self, request, *args, **kwargs):
        form = SecanteForm(request.POST)
        context = self.get_context_data()
        if form.is_valid():
            x0 = form.cleaned_data['x0']
            x1 = form.cleaned_data['x1']
            tol = form.cleaned_data['tol']
            n = form.cleaned_data['n']
            fun_str = form.cleaned_data['fun']
            export = form.cleaned_data['export']

            x = sp.symbols('x')
            f = sp.lambdify(x, sp.sympify(fun_str), 'numpy')

            def secante(x0, x1, tol, n, f):
                fx0 = f(x0)
                if fx0 == 0:
                    return x0
                else:
                    fx1 = f(x1)
                    i = 0
                    error = tol + 1
                    den = fx1 - fx0
                    while error > tol and fx1 != 0 and den != 0 and i < n:
                        x2 = x1 - ((fx1 * (x1 - x0)) / den)
                        error = np.abs(x2 - x1)
                        x0 = x1
                        fx0 = fx1
                        x1 = x2
                        fx1 = f(x1)
                        den = fx1 - fx0
                        i += 1
                    if fx1 == 0:
                        return [x1, "Error de " + str(0)]
                    elif error < tol:
                        return [x1, "Error de " + str(error)]
                    elif den == 0:
                        return ["Posible raíz múltiple"]
                    else:
                        return [x1, "Fracasó en " + str(n) + " iteraciones"]

            result = secante(x0, x1, tol, n, f)
            
            if export:
                response = HttpResponse(content_type='text/plain')
                response['Content-Disposition'] = 'attachment; filename="resultados_secante.txt"'
                response.write(f"Raíz aproximada: {result[0]}\n")
                response.write(f"{result[1]}\n")
                return response

            context['result'] = result
            context['form'] = form
        else:
            context['form'] = form

        return render(request, self.template_name, context)


#------------Raices_Multiples-----------------------------

class RaicesMultiplesForm(forms.Form):
    x0 = forms.FloatField(label='X0', required=True)
    tol = forms.FloatField(label='Tol', required=True)
    n = forms.IntegerField(label='Niter', required=True)
    fun = forms.CharField(label='Function', required=True, widget=forms.TextInput(attrs={'placeholder': 'Ingrese la función en términos de x'}))
    deriv1 = forms.CharField(label='First Derivative', required=True, widget=forms.TextInput(attrs={'placeholder': 'Ingrese la primera derivada en términos de x'}))
    deriv2 = forms.CharField(label='Second Derivative', required=True, widget=forms.TextInput(attrs={'placeholder': 'Ingrese la segunda derivada en términos de x'}))
    export = forms.BooleanField(label='Exportar resultados a TXT', required=False)
class RaicesMultiplesPageView(TemplateView):
    template_name = 'raicesmultiples.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = RaicesMultiplesForm()
        return context

    def post(self, request, *args, **kwargs):
        form = RaicesMultiplesForm(request.POST)
        context = self.get_context_data()
        if form.is_valid():
            x0 = form.cleaned_data['x0']
            tol = form.cleaned_data['tol']
            n = form.cleaned_data['n']
            fun_str = form.cleaned_data['fun']
            deriv1_str = form.cleaned_data['deriv1']
            deriv2_str = form.cleaned_data['deriv2']
            export = form.cleaned_data['export']

            x = sp.symbols('x')
            f = sp.lambdify(x, sp.sympify(fun_str), 'numpy')
            derivada1 = sp.lambdify(x, sp.sympify(deriv1_str), 'numpy')
            derivada2 = sp.lambdify(x, sp.sympify(deriv2_str), 'numpy')

            def RaicesMul(x0, tol, n, f, derivada1, derivada2):
                fx0 = f(x0)
                if fx0 == 0:
                    return x0
                else:
                    fx0 = f(x0)
                    fpx0 = derivada1(x0)
                    fppx0 = derivada2(x0)
                    i = 0
                    error = tol + 1
                    den = (fpx0**2) - (fx0 * fppx0)
                    while error > tol and fx0 != 0 and den != 0 and i < n:
                        x1 = x0 - ((fx0 * fpx0) / den)
                        fx0 = f(x1)
                        fpx0 = derivada1(x1)
                        fppx0 = derivada2(x1)
                        den = (fpx0**2) - (fx0 * fppx0)
                        error = np.abs(x1 - x0)
                        x0 = x1
                        i += 1
                    if fx0 == 0:
                        return x0
                    elif error < tol:
                        return [x0, "Error de " + str(error)]
                    else:
                        return [x0, "Fracasó en " + str(n) + " iteraciones"]

            result = RaicesMul(x0, tol, n, f, derivada1, derivada2)
            
            if export:
                response = HttpResponse(content_type='text/plain')
                response['Content-Disposition'] = 'attachment; filename="resultados_raices_multiples.txt"'
                response.write(f"Raíz aproximada: {result[0]}\n")
                response.write(f"{result[1]}\n")
                return response

            context['result'] = result
            context['form'] = form
        else:
            context['form'] = form

        return render(request, self.template_name, context)
    
    
#------------------GaussSeidel----------------------------- 
class GaussSeidelForm(forms.Form):
    rows = forms.IntegerField(label='Número de filas (n)', min_value=1, required=True)
    cols = forms.IntegerField(label='Número de columnas (n)', min_value=1, required=True)
    tol = forms.FloatField(label='Tolerancia', required=True)
    numIter = forms.IntegerField(label='Número máximo de iteraciones', required=True)
    export = forms.BooleanField(label='Exportar resultados a TXT', required=False)
    A = forms.CharField(widget=forms.HiddenInput(), required=False)
    b = forms.CharField(widget=forms.HiddenInput(), required=False)
    
class GaussSeidelPageView(TemplateView):
    template_name = 'gaussseidel.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = GaussSeidelForm()
        return context

    def post(self, request, *args, **kwargs):
        form = GaussSeidelForm(request.POST)
        context = self.get_context_data()
        if form.is_valid():
            rows = form.cleaned_data['rows']
            cols = form.cleaned_data['cols']
            tol = form.cleaned_data['tol']
            numIter = form.cleaned_data['numIter']
            export = form.cleaned_data['export']

            
            A = [[float(request.POST[f'A_{i}_{j}']) for j in range(cols)] for i in range(rows)]
            b = [float(request.POST[f'b_{i}']) for i in range(rows)]

            A = np.array(A)
            b = np.array(b).reshape(-1, 1)

            def GaussSeidel(A, b, tol, numIter):
                n = np.size(A, 0)
                L = -np.tril(A, -1)
                U = -np.triu(A, 1)
                D = A + L + U
                x0 = np.zeros([n, 1])
                Tg = np.matmul(inv(D - L), U)
                autoval, autovec = np.linalg.eig(Tg)
                autoval = abs(autoval)

                for lam in autoval:
                    if lam >= 1:
                        return ["El método no pudo converger de acuerdo a los parámetros ingresados"], []

                C = np.matmul(inv(D - L), b)
                xn = np.matmul(Tg, x0) + C
                error = np.amax(abs(xn - (np.dot(Tg, xn) + C)))
                error = np.amax(error)
                iter = 0
                iteraciones = []

                while ((error > tol) and (iter < numIter)):
                    nuevo = np.matmul(Tg, xn) + C
                    error = np.amax(abs(nuevo - xn))
                    error = np.amax(error)
                    iteraciones.append((iter, xn.flatten().tolist(), nuevo.flatten().tolist(), error))
                    xn = nuevo
                    iter = iter + 1

                iteraciones.append((iter, xn.flatten().tolist(), nuevo.flatten().tolist(), error))
                return xn.flatten().tolist(), iteraciones

            result, iteraciones = GaussSeidel(A, b, tol, numIter)

            if export:
                response = HttpResponse(content_type='text/plain')
                response['Content-Disposition'] = 'attachment; filename="resultados_gaussseidel.txt"'
                response.write(f"Solución aproximada: {result}\n")
                response.write("Iteración\tXanterior\tXnuevo\tError\n")
                for row in iteraciones:
                    iter_num, xant, xnuevo, err = row
                    response.write(f"{iter_num}\t{xant}\t{xnuevo}\t{err}\n")
                return response

            context['result'] = result
            context['iterations'] = iteraciones
            context['form'] = form
        else:
            context['form'] = form

        return render(request, self.template_name, context)
    
    
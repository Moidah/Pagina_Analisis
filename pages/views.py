from django.shortcuts import render, HttpResponse
from django import forms
from django.views.generic import TemplateView
import plotly.graph_objs as go
from plotly.offline import plot
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import io
import urllib, base64

#-------------Para ingresar todos los parametros de los metodos--------------------------
class FunctionForm(forms.Form):
    function = forms.CharField(label='Función', widget=forms.TextInput(attrs={'placeholder': 'Ingrese la función en términos de x'}))
#---------------------------------------------------------------------------------------
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
            
            function_expr = sp.sympify(fun)
            x_values = np.linspace(xi, xs, 400)
            y_values = [function_expr.subs(x, val) for val in x_values]

            fig, ax = plt.subplots(figsize=(10, 6))  # Aumenta el tamaño de la figura
            ax.plot(x_values, y_values, label=f'f(x) = {fun}', color='blue')
            ax.axhline(0, color='black', linewidth=0.5)
            ax.axvline(0, color='black', linewidth=0.5)
            ax.grid(color='gray', linestyle='--', linewidth=0.5)
            ax.set_xlabel('x')  # Etiqueta del eje x
            ax.set_ylabel('f(x)')  # Etiqueta del eje y
            ax.set_title('Gráfica de la función')  # Título de la gráfica
            plt.legend()

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

            # Guardar la figura en un buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            string = base64.b64encode(buf.read())
            uri = 'data:image/png;base64,' + urllib.parse.quote(string)

            context['result'] = result
            context['iterations'] = iterations
            context['form'] = form
            context['image'] = uri

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
    xi = forms.FloatField(label="Xi")
    xs = forms.FloatField(label="Xs")
    tol = forms.FloatField(label="Tolerancia")
    n = forms.IntegerField(label="Número de Iteraciones")
    fun = forms.CharField(label="Función", max_length=100)
    export = forms.BooleanField(label="Exportar resultados a TXT", required=False)
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
                    while error > tol and abs(fxm) > tol and fxm != 0 and i < n:
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

#----------NEWTON----------------------
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
                    fact = f.subs(x, xact)
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
            try:
                x0 = float(form.cleaned_data['x0'])
                x1 = float(form.cleaned_data['x1'])
                tol = float(form.cleaned_data['tol'])
                n = int(form.cleaned_data['n'])
                fun_str = form.cleaned_data['fun']
                export = form.cleaned_data['export']
            except ValueError:
                context['error'] = "Asegúrate de que los valores numéricos están bien formateados."
                context['form'] = form
                return render(request, self.template_name, context)

            x = sp.symbols('x')
            try:
                f = sp.lambdify(x, sp.sympify(fun_str), 'numpy')
            except sp.SympifyError:
                context['error'] = "Error al parsear las expresiones. Asegúrate de usar la sintaxis correcta."
                context['form'] = form
                return render(request, self.template_name, context)

            def secante(x0, x1, tol, n, f):
                fx0 = f(x0)
                if fx0 == 0:
                    return x0, []
                else:
                    fx1 = f(x1)
                    i = 0
                    error = tol + 1
                    den = fx1 - fx0
                    table = []
                    while error > tol and fx1 != 0 and den != 0 and i < n:
                        x2 = x1 - ((fx1 * (x1 - x0)) / den)
                        error = np.abs(x2 - x1)
                        table.append((i, x0, x1, x2, fx1, error))
                        x0 = x1
                        fx0 = fx1
                        x1 = x2
                        fx1 = f(x1)
                        den = fx1 - fx0
                        i += 1
                    table.append((i, x0, x1, x2, fx1, error))
                    if fx1 == 0:
                        return [x1, "Error de " + str(0)], table
                    elif error < tol:
                        return [x1, "Error de " + str(error)], table
                    elif den == 0:
                        return ["Posible raíz múltiple"], table
                    else:
                        return [x1, "Fracasó en " + str(n) + " iteraciones"], table

            result, table = secante(x0, x1, tol, n, f)

            if export:
                response = HttpResponse(content_type='text/plain')
                response['Content-Disposition'] = 'attachment; filename="resultados_secante.txt"'
                response.write(f"Raíz aproximada: {result[0]}\n")
                response.write(f"{result[1]}\n")
                response.write("Iteración\tX0\tX1\tX2\tf(X1)\tError\n")
                for row in table:
                    response.write("\t".join(map(str, row)) + "\n")
                return response

            context['result'] = result
            context['table'] = table
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
            try:
                x0 = float(form.cleaned_data['x0'])
                tol = float(form.cleaned_data['tol'])
                n = int(form.cleaned_data['n'])
                fun_str = form.cleaned_data['fun']
                deriv1_str = form.cleaned_data['deriv1']
                deriv2_str = form.cleaned_data['deriv2']
                export = form.cleaned_data['export']
            except ValueError:
                context['error'] = "Asegúrate de que los valores numéricos están bien formateados."
                context['form'] = form
                return render(request, self.template_name, context)

            x = sp.symbols('x')
            try:
                f = sp.sympify(fun_str.replace('^', '**'))
                deriv1 = sp.sympify(deriv1_str.replace('^', '**'))
                deriv2 = sp.sympify(deriv2_str.replace('^', '**'))
            except sp.SympifyError:
                context['error'] = "Error al parsear las expresiones. Asegúrate de usar la sintaxis correcta."
                context['form'] = form
                return render(request, self.template_name, context)

            def RaicesMul(x0, tol, n, f, deriv1, deriv2):
                fx0 = f.subs(x, x0)
                table = []
                if fx0 == 0:
                    return x0, table
                else:
                    fpx0 = deriv1.subs(x, x0)
                    fppx0 = deriv2.subs(x, x0)
                    iter = 0
                    error = tol + 1
                    den = (fpx0**2) - (fx0 * fppx0)
                    while error > tol and fx0 != 0 and den != 0 and iter < n:
                        x1 = x0 - (fx0 * fpx0) / den
                        fx0 = f.subs(x, x1)
                        fpx0 = deriv1.subs(x, x1)
                        fppx0 = deriv2.subs(x, x1)
                        den = (fpx0**2) - (fx0 * fppx0)
                        error = abs(x1 - x0)
                        table.append((iter, x0, x1, fx0, error))
                        x0 = x1
                        iter += 1
                    table.append((iter, x0, x1, fx0, error))
                    if fx0 == 0:
                        return x0, table
                    elif error < tol:
                        return x0, table
                    else:
                        return x0, table

            result, table = RaicesMul(x0, tol, n, f, deriv1, deriv2)

            if export:
                response = HttpResponse(content_type='text/plain')
                response['Content-Disposition'] = 'attachment; filename="resultados_raicesmultiples.txt"'
                response.write(f"Raíz aproximada: {result}\n")
                response.write("Iteración\tX0\tX1\tf(X1)\tError\n")
                for row in table:
                    response.write("\t".join(map(str, row)) + "\n")
                return response

            context['result'] = result
            context['table'] = table
            context['form'] = form
        else:
            context['form'] = form

        return render(request, self.template_name, context)
    
    
#------------------GaussSeidel----------------------------- 
class GaussSeidelForm(forms.Form):
    rows = forms.IntegerField(label="Número de filas")
    cols = forms.IntegerField(label="Número de columnas")
    tol = forms.FloatField(label="Tolerancia")
    niter = forms.IntegerField(label="Número máximo de iteraciones")
    export = forms.BooleanField(required=False, label="Exportar resultados a TXT")
    
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
            n = form.cleaned_data['rows']
            tol = form.cleaned_data['tol']
            niter = form.cleaned_data['niter']
            A = np.zeros((n, n))
            b = np.zeros(n)
            x0 = np.zeros(n)

            try:
                for i in range(n):
                    for j in range(n):
                        A[i][j] = float(request.POST.get(f'A_{i}_{j}', 0))
                    b[i] = float(request.POST.get(f'b_{i}', 0))
                    x0[i] = float(request.POST.get(f'x0_{i}', 0))
            except ValueError as e:
                context['error'] = "Asegúrate de ingresar todos los valores de la matriz y el vector correctamente."
                context['form'] = form
                return render(request, self.template_name, context)

            def gauss_seidel(A, b, x0, n, tol, niter):
                # Check for zero in the diagonal
                if any(A[i][i] == 0 for i in range(n)):
                    return None, "El método no puede converger debido a un 0 en la diagonal de la matriz A."

                iteraciones = []
                it = 0
                t = tol + 1
                iteraciones.append([it, x0.copy().tolist(), t])
                while t > tol and it < niter:
                    it += 1
                    x_nuevo = np.zeros(n)
                    for j in range(n):
                        sum_ax = sum(A[j][k] * x_nuevo[k] if k < j else A[j][k] * x0[k] for k in range(n) if k != j)
                        x_nuevo[j] = (b[j] - sum_ax) / A[j][j]
                    t = np.max(np.abs(x0 - x_nuevo))
                    x0 = x_nuevo
                    iteraciones.append([it, x0.copy().tolist(), t])
                return iteraciones, x0.tolist()

            iterations, result = gauss_seidel(A, b, x0, n, tol, niter)

            if iterations is None:
                context['error'] = result
                context['form'] = form
                return render(request, self.template_name, context)

            if form.cleaned_data['export']:
                response = HttpResponse(content_type='text/plain')
                response['Content-Disposition'] = 'attachment; filename="resultados_gaussseidel.txt"'
                response.write("Iteración\tVector\tTolerancia\n")
                for row in iterations:
                    response.write(f"{row[0]}\t{row[1]}\t{row[2]}\n")
                return response

            context['result'] = result
            context['iterations'] = iterations
            context['form'] = form
        else:
            context['form'] = form

        return render(request, self.template_name, context)

    
#---------------------Jacobi------------------------------------------------
class JacobiForm(forms.Form):
    n = forms.IntegerField(label='Dimensión de la matriz (n)', required=True)
    tol = forms.FloatField(label='Tolerancia', required=True)
    niter = forms.IntegerField(label='Número máximo de iteraciones', required=True)
    export = forms.BooleanField(label='Exportar resultados a TXT', required=False)

class JacobiPageView(TemplateView):
    template_name = 'jacobi.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = JacobiForm()
        return context

    def post(self, request, *args, **kwargs):
        form = JacobiForm(request.POST)
        context = self.get_context_data()
        if form.is_valid():
            n = form.cleaned_data['n']
            tol = form.cleaned_data['tol']
            niter = form.cleaned_data['niter']
            A = np.zeros((n, n))
            b = np.zeros(n)
            x0 = np.zeros(n)

            for i in range(n):
                for j in range(n):
                    value = request.POST.get(f'A_{i}_{j}')
                    if value is not None and value != '':
                        A[i][j] = float(value)
                    else:
                        context['error'] = "Todos los valores de la matriz A deben ser proporcionados."
                        return render(request, self.template_name, context)

                value = request.POST.get(f'b_{i}')
                if value is not None and value != '':
                    b[i] = float(value)
                else:
                    context['error'] = "Todos los valores del vector b deben ser proporcionados."
                    return render(request, self.template_name, context)

                value = request.POST.get(f'x0_{i}')
                if value is not None and value != '':
                    x0[i] = float(value)
                else:
                    context['error'] = "Todos los valores del vector x0 deben ser proporcionados."
                    return render(request, self.template_name, context)

            # Verificar ceros en la diagonal de la matriz
            if any(A[i][i] == 0 for i in range(n)):
                context['error'] = "El método no puede converger debido a un 0 en la diagonal de la matriz A."
                return render(request, self.template_name, context)

            def jacobi(A, b, x0, n, tol, niter):
                iteraciones = []
                it = 0
                t = tol + 1
                iteraciones.append([it, x0.copy().tolist(), t])
                while t > tol and it < niter:
                    it += 1
                    x_nuevo = np.zeros(n)
                    for j in range(n):
                        suma = sum(-A[j][k] * x0[k] if j != k else 0 for k in range(n))
                        x_nuevo[j] = (b[j] + suma) / A[j][j]
                    t = np.max(np.abs(x0 - x_nuevo))
                    x0 = x_nuevo
                    iteraciones.append([it, x0.copy().tolist(), t])
                return iteraciones, x0.tolist()

            iteraciones, result = jacobi(A, b, x0, n, tol, niter)

            if form.cleaned_data['export']:
                response = HttpResponse(content_type='text/plain')
                response['Content-Disposition'] = 'attachment; filename="resultados_jacobi.txt"'
                response.write("Iteración\tVector\tTolerancia\n")
                for row in iteraciones:
                    response.write(f"{row[0]}\t{row[1]}\t{row[2]}\n")
                return response

            context['result'] = result
            context['iterations'] = iteraciones
            context['form'] = form
        else:
            context['form'] = form

        return render(request, self.template_name, context)


    
#-------------SOR--------------------------------------------------------
class SORForm(forms.Form):
    n = forms.IntegerField(label="Dimensiones de la matriz (n)")
    tol = forms.FloatField(label="Tolerancia")
    niter = forms.IntegerField(label="Número máximo de iteraciones")
    w = forms.FloatField(label="Parámetro de relajación (ω)")
    export = forms.BooleanField(label="Exportar resultados a TXT", required=False)
    

class SORPageView(TemplateView):
    template_name = 'sor.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = SORForm()
        return context

    def post(self, request, *args, **kwargs):
        form = SORForm(request.POST)
        context = self.get_context_data()
        if form.is_valid():
            n = form.cleaned_data['n']
            tol = form.cleaned_data['tol']
            niter = form.cleaned_data['niter']
            w = form.cleaned_data['w']
            A = np.zeros((n, n))
            b = np.zeros(n)
            x0 = np.zeros(n)

            for i in range(n):
                for j in range(n):
                    value = request.POST.get(f'A_{i}_{j}')
                    if value is not None and value != '':
                        A[i][j] = float(value)
                    else:
                        context['error'] = "Todos los valores de la matriz A deben ser proporcionados."
                        return render(request, self.template_name, context)

                value = request.POST.get(f'b_{i}')
                if value is not None and value != '':
                    b[i] = float(value)
                else:
                    context['error'] = "Todos los valores del vector b deben ser proporcionados."
                    return render(request, self.template_name, context)

                value = request.POST.get(f'x0_{i}')
                if value is not None and value != '':
                    x0[i] = float(value)
                else:
                    context['error'] = "Todos los valores del vector x0 deben ser proporcionados."
                    return render(request, self.template_name, context)

            # Verificar ceros en la diagonal de la matriz
            if any(A[i][i] == 0 for i in range(n)):
                context['error'] = "El método no puede converger debido a un 0 en la diagonal de la matriz A."
                return render(request, self.template_name, context)

            def sor(A, b, x0, n, tol, w):
                iteraciones = []
                it = 0
                t = tol + 1
                iteraciones.append([it, x0.copy().tolist(), t])
                while t > tol and it < niter:
                    it += 1
                    x_nuevo = np.zeros(n)
                    for j in range(n):
                        sum_ax = sum(A[j][k] * x_nuevo[k] if k < j else A[j][k] * x0[k] for k in range(n) if k != j)
                        x_nuevo[j] = (b[j] - sum_ax) / A[j][j]
                        x_nuevo[j] = (1 - w) * x0[j] + w * x_nuevo[j]
                    t = np.max(np.abs(x0 - x_nuevo))
                    x0 = x_nuevo
                    iteraciones.append([it, x0.copy().tolist(), t])
                return iteraciones, x0.tolist()

            iterations, result = sor(A, b, x0, n, tol, w)

            if form.cleaned_data['export']:
                response = HttpResponse(content_type='text/plain')
                response['Content-Disposition'] = 'attachment; filename="resultados_sor.txt"'
                response.write("Iteración\tVector\tTolerancia\n")
                for row in iterations:
                    response.write(f"{row[0]}\t{row[1]}\t{row[2]}\n")
                return response

            context['result'] = result
            context['iterations'] = iterations
            context['form'] = form
        else:
            context['form'] = form

        return render(request, self.template_name, context)



#------------------SORMatricial-----------------------------------  
class SORMForm(forms.Form):
    n = forms.IntegerField(label="Dimensiones de la matriz (n)")
    tol = forms.FloatField(label="Tolerancia")
    niter = forms.IntegerField(label="Número máximo de iteraciones")
    w = forms.FloatField(label="Parámetro de relajación (ω)")
    export = forms.BooleanField(label="Exportar resultados a TXT", required=False) 
    
class SORMatricialPageView(TemplateView):
    template_name = 'sor_matricial.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = SORMForm()
        return context

    def post(self, request, *args, **kwargs):
        form = SORMForm(request.POST)
        context = self.get_context_data()
        if form.is_valid():
            n = form.cleaned_data['n']
            tol = form.cleaned_data['tol']
            niter = form.cleaned_data['niter']
            w = form.cleaned_data['w']
            A = np.zeros((n, n))
            b = np.zeros(n)
            x0 = np.zeros(n)

            for i in range(n):
                for j in range(n):
                    A[i][j] = float(request.POST.get(f'A_{i}_{j}'))
                b[i] = float(request.POST.get(f'b_{i}'))
                x0[i] = float(request.POST.get(f'x0_{i}'))

            def sor_matricial(A, b, x0, n, tol, w):
                table = []
                it = 0
                t = tol + 1
                table.append([it, x0.copy().tolist(), t])
                D = np.diag(np.diag(A))
                L = (-1) * np.tril(A - D)
                U = (-1) * np.triu(A - D)
                T = np.linalg.inv(D - w * L) @ ((1 - w) * D + w * U)
                C = w * np.linalg.inv(D - w * L) @ np.transpose(b)
                while t > tol and it < niter:
                    it += 1
                    x_nuevo = np.transpose(T @ np.transpose(x0) + C)
                    t = max(abs(x0 - x_nuevo))
                    x0 = x_nuevo
                    table.append([it, x0.copy().tolist(), t])
                return table, x0.tolist()

            iterations, result = sor_matricial(A, b, x0, n, tol, w)

            if form.cleaned_data['export']:
                response = HttpResponse(content_type='text/plain')
                response['Content-Disposition'] = 'attachment; filename="resultados_sor_matricial.txt"'
                response.write("Iteración\tVector\tTolerancia\n")
                for row in iterations:
                    response.write(f"{row[0]}\t{row[1]}\t{row[2]}\n")
                return response

            context['result'] = result
            context['iterations'] = iterations
            context['form'] = form
        else:
            context['form'] = form

        return render(request, self.template_name, context)

#-------------------Vandermonde----------------------------

class VandermondeForm(forms.Form):
    n = forms.IntegerField(label="Número de puntos")
    export = forms.BooleanField(label="Exportar resultados a TXT", required=False)

class VandermondePageView(TemplateView):
    template_name = 'vandermonde.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = VandermondeForm()
        return context

    def post(self, request, *args, **kwargs):
        form = VandermondeForm(request.POST)
        context = self.get_context_data()
        if form.is_valid():
            n = form.cleaned_data['n']
            xs = [float(request.POST.get(f'xs_{i}')) for i in range(n)]
            y = [float(request.POST.get(f'y_{i}')) for i in range(n)]

            def vandermonde(xs, y, n):
                x = sp.symbols('x')
                A = np.array([[i**j for j in range(n-1, -1, -1)] for i in xs])
                C = np.linalg.inv(A) @ np.transpose(y)
                C = C[::-1]
                expr = 0
                funcion = "F(x)="
                for i in range(n-1, -1, -1):
                    if i == 0:
                        expr += C[i]
                        funcion += str(C[i])
                        break
                    expr += C[i] * x**i
                    funcion += str(C[i]) + "x^" + str(i) + "+"
                return funcion, expr

            funcion, expr = vandermonde(xs, y, n)
            context['funcion'] = funcion
            context['expr'] = expr
            context['form'] = form
        else:
            context['form'] = form

        return render(request, self.template_name, context)
    
    
#---------Newton_Interpolante--------------------------

class NewtonInterpolanteForm(forms.Form):
    n = forms.IntegerField(label="Número de puntos")
    export = forms.BooleanField(label="Exportar resultados a TXT", required=False)

class NewtonInterpolantePageView(TemplateView):
    template_name = 'newton_interpolante.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = NewtonInterpolanteForm()
        return context

    def post(self, request, *args, **kwargs):
        form = NewtonInterpolanteForm(request.POST)
        context = self.get_context_data()
        if form.is_valid():
            n = form.cleaned_data['n']
            xs = [float(request.POST.get(f'xs_{i}')) for i in range(n)]
            y = [float(request.POST.get(f'y_{i}')) for i in range(n)]

            def newton_interpolante(xs, y, n):
                x = sp.symbols('x')
                dif_div = np.zeros((n, n + 1))
                dif_div[:, 0] = xs
                dif_div[:, 1] = y
                expr = 0
                for i in range(n - 1):
                    for j in range(n - 1):
                        if i <= j:
                            dif_div[j + 1][i + 2] = (dif_div[j + 1][i + 1] - dif_div[j][i + 1]) / (dif_div[j + 1][0] - dif_div[j + 1 - i - 1][0])
                for i in range(n):
                    aux = dif_div[i][i + 1]
                    for j in range(i):
                        aux *= (x - dif_div[j][0])
                    expr += aux
                expr = sp.expand(expr)
                funcion = f"F(x) = {expr}"
                return funcion, expr

            funcion, expr = newton_interpolante(xs, y, n)
            context['funcion'] = funcion
            context['expr'] = expr
            context['form'] = form
        else:
            context['form'] = form

        return render(request, self.template_name, context)
    
#--------lagrange---------------------------------------------------

class LagrangeForm(forms.Form):
    n = forms.IntegerField(label="Número de puntos")
    export = forms.BooleanField(label="Exportar resultados a TXT", required=False)
    
class LagrangePageView(TemplateView):
    template_name = 'lagrange.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = LagrangeForm()
        return context

    def post(self, request, *args, **kwargs):
        form = LagrangeForm(request.POST)
        context = self.get_context_data()
        if form.is_valid():
            n = form.cleaned_data['n']
            xs = [float(request.POST.get(f'xs_{i}')) for i in range(n)]
            y = [float(request.POST.get(f'y_{i}')) for i in range(n)]

            def lagrange(xs, y, n):
                x = sp.symbols('x')
                expr = 0
                for i in range(n):
                    aux = 1
                    for j in range(n):
                        if i != j:
                            aux *= (x - xs[j]) / (xs[i] - xs[j])
                    expr += aux * y[i]
                expr = sp.expand(expr)
                funcion = f"F(x) = {expr}"
                return funcion, expr

            expr = lagrange(xs, y, n)
            context['expr'] = expr
            context['form'] = form

            if form.cleaned_data['export']:
                response = HttpResponse(content_type='text/plain')
                response['Content-Disposition'] = 'attachment; filename="resultados_lagrange.txt"'
                response.write(f"Polinomio Interpolante:\n{expr}\n")
                return response
        else:
            context['form'] = form

        return render(request, self.template_name, context)
    
#----------------------spline-----------------------------------

class SplineForm(forms.Form):
    n = forms.IntegerField(label='Número de puntos', min_value=2, required=True)
    xs = forms.CharField(label='Valores de X (separados por comas)', required=True)
    y = forms.CharField(label='Valores de Y (separados por comas)', required=True)
    

class SplineLinealPageView(TemplateView):
    template_name = 'spline.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = SplineForm()
        return context

    def post(self, request, *args, **kwargs):
        form = SplineForm(request.POST)
        context = self.get_context_data()
        if form.is_valid():
            n = form.cleaned_data['n']
            xs = np.array([float(x.strip()) for x in form.cleaned_data['xs'].split(',')])
            y = np.array([float(y.strip()) for y in form.cleaned_data['y'].split(',')])

            coef, polinomios = self.spline_lineal(xs, y, n)

            context['coeficientes'] = coef.tolist()
            context['polinomios'] = polinomios
        context['form'] = form
        return render(request, self.template_name, context)

    def spline_lineal(self, xs, y, n):
        coef = np.zeros((n-1, 2))
        polinomios = []
        for i in range(n-1):
            A = np.array([[xs[i], 1], [xs[i+1], 1]])
            B = np.array([[y[i]], [y[i+1]]])
            coef[i, :] = np.transpose(np.linalg.inv(A) @ B)
            polinomios.append(f"P{i+1}(x) = {round(coef[i][0], 4)}x + {round(coef[i][1], 4)}")
        return coef, polinomios
from django.shortcuts import render
from django import forms
from django.views.generic import TemplateView
import plotly.graph_objs as go
from plotly.offline import plot
import numpy as np
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

            x = sp.symbols('x')
            fi = sp.sympify(fun).subs(x, xi)
            fs = sp.sympify(fun).subs(x, xs)
            fm = []
            E = []

            if fi == 0:
                s = xi
                E.append(0)
                result = f"{xi} es raíz de f(x)"
            elif fs == 0:
                s = xs
                E.append(0)
                result = f"{xs} es raíz de f(x)"
            elif fs * fi < 0:
                c = 0
                Xm = (xi + xs) / 2
                fe = sp.sympify(fun).subs(x, Xm)
                fm.append(fe)
                E.append(100)

                while E[c] > tol and fe != 0 and c < niter:
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

            context['result'] = result
            context['iterations'] = list(zip(fm, E))
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

            x = sp.symbols('x')
            g = sp.sympify(g_str)
            f = (x)**2 -100

            def Cre_o_Decre(f, x0):
                return sp.diff(f, x).subs(x, x0) > 0

            def PuntoFijo(g, x0, tol, Nmax):
                
                #Inicialización
                xant = x0
                E = 1000
                cont = 0
                iteraciones = []
                
                
                #Ciclo
                while E > tol and cont < Nmax:
                    xact = g.subs(x, xant)
                    E = abs(xact - xant)
                    cont += 1
                    iteraciones.append((cont, xact, E))
                    xant = xact

                return [xact, cont, E], iteraciones

            creciente = Cre_o_Decre(f, x0)
            (xact, cont, E), iteraciones = PuntoFijo(g, x0, tol, Nmax)
            result = f"Convergió a {xact} con una tolerancia de {tol}" if E < tol else f"No convergió después de {Nmax} iteraciones"

            context['creciente'] = "La función es creciente en el punto inicial" if creciente else "La función es decreciente en el punto inicial"
            context['result'] = result
            context['iterations'] = iteraciones
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

            x = sp.symbols('x')
            fun = sp.lambdify(x, sp.sympify(fun_str), 'numpy')

            def ReglaF(xi, xs, tol, n, f):
                fxi = f(xi)
                fxs = f(xs)
                if fxi == 0: 
                    return xi 
                elif fxs == 0:
                    return xs
                elif fxi * fxs < 0: 
                    xm = xi - ((fxi * (xs - xi))) / (fxs - fxi)
                    fxm = f(xm)
                    i = 1 
                    error = tol + 1 
                    while error > tol and fxm != 0 and i < n:
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
                    if fxm == 0:
                        return [xm, "Error de " + str(0)]
                    elif error < tol:
                        return [xm, "Error de " + str(error)]
                    else:
                        return ["Fracasó en " + str(n) + " iteraciones"]
                else:
                    return ["Intervalo inadecuado"]

            result = ReglaF(xi, xs, tol, n, fun)

            context['result'] = result
            context['form'] = form
        else:
            context['form'] = form

        return render(request, self.template_name, context)
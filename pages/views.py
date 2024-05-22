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
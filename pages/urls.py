from django.urls import path
from .views import HomePageView, MetodosPageView, NosotrosPageView, BiseccionPageView, PuntoFijoPageView, ReglaFalsaPageView, NewtonPageView, SecantePageView, RaicesMultiplesPageView, GaussSeidelPageView, JacobiPageView, SORPageView, SORMatricialPageView, VandermondePageView, NewtonInterpolantePageView, LagrangePageView, SplineLinealPageView


urlpatterns = [
    path('', HomePageView.as_view(), name='home'),
    path('metodos/', MetodosPageView.as_view(), name='metodos'),
    path('nosotros/', NosotrosPageView.as_view(), name='nosotros'),
    path('biseccion/', BiseccionPageView.as_view(), name='biseccion'),
    path('puntofijo/', PuntoFijoPageView.as_view(), name='puntofijo'),
    path('reglafalsa/', ReglaFalsaPageView.as_view(), name='reglafalsa'),
    path('newton/', NewtonPageView.as_view(), name='newton'),
    path('secante/', SecantePageView.as_view(), name='secante'),
    path('raicesmultiples/', RaicesMultiplesPageView.as_view(), name='raicesmultiples'),
    path('gaussseidel/', GaussSeidelPageView.as_view(), name='gaussseidel'),
    path('jacobi/', JacobiPageView.as_view(), name='jacobi'),
    path('sor/', SORPageView.as_view(), name='sor'),
    path('sor_matricial/', SORMatricialPageView.as_view(), name='sor_matricial'),
    path('vandermonde/', VandermondePageView.as_view(), name='vandermonde'),
    path('vandermonde/', VandermondePageView.as_view(), name='vandermonde'),
    path('newton_interpolante/', NewtonInterpolantePageView.as_view(), name='newton_interpolante'),
    path('lagrange/', LagrangePageView.as_view(), name='lagrange'),
    path('spline/', SplineLinealPageView.as_view(), name='spline'),
]

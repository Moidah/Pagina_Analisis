from django.urls import path
from .views import HomePageView, MetodosPageView, NosotrosPageView, BiseccionPageView, PuntoFijoPageView, ReglaFalsaPageView, NewtonPageView, SecantePageView, RaicesMultiplesPageView, GaussSeidelPageView


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
]

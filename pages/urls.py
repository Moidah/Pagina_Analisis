from django.urls import path
from .views import HomePageView, MetodosPageView, NosotrosPageView, BiseccionPageView


urlpatterns = [
    path('', HomePageView.as_view(), name='home'),
    path('metodos/', MetodosPageView.as_view(), name='metodos'),
    path('nosotros/', NosotrosPageView.as_view(), name='nosotros'),
    path('biseccion/', BiseccionPageView.as_view(), name='biseccion'),
]

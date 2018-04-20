from django.urls import path
from . import views

app_name = 'extender'
urlpatterns = [
    path('', views.index, name='index'),
    path('search', views.search, name='search'),
    path('search/<str:term>/', views.search, name='search'),
    path('visualize', views.visualize, name='visualize'),
    path('evaluator', views.evaluator, name='evaluator'),
]
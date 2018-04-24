from django.urls import path
from . import views

app_name = 'extender'
urlpatterns = [
    path('', views.index, name='index'),
    path('search_relevant_terms', views.search_relevant_terms, name='search_relevant_terms'),
    path('search_relevant_terms/<str:term>/', views.search_relevant_terms, name='search_relevant_terms'),
    path('search_relevant_titles', views.search_relevant_titles, name='search_relevant_titles'),
    path('search_relevant_titles/<str:title>/', views.search_relevant_titles, name='search_relevant_titles'),
    path('search_relevant_titles/<str:title>/<int:next>', views.search_relevant_titles, name='search_relevant_titles'),

    path('visualize', views.visualize, name='visualize'),
    path('evaluator', views.evaluator, name='evaluator'),
]
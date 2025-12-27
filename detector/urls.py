# detector/urls.py

from django.urls import path
from . import views

urlpatterns = [
    # The root path for this app (which is your website's homepage)
    path('', views.detection_view, name='home'), 
]
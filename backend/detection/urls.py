from django.urls import path
from . import views

urlpatterns = [
    path("", views.api_home, name="api_home"),
    path("upload/", views.upload_leaf, name="upload_leaf"),

    path(
        "history/",
        views.PredictionHistoryListView.as_view(),
        name="prediction_history",
    ),

    path(
        "weather/",
        views.weather_by_city,
        name="weather_by_city",
    ),
]
from django.urls import path
from . import views

urlpatterns = [
    path("", views.api_home, name="api_home"),
    path("upload/", views.upload_leaf, name="upload_leaf"),
    path("signup/", views.signup_user, name="signup_user"),
    path("login/", views.login_user, name="login_user"),
    path("logout/", views.logout_user, name="logout_user"),
    path("me/", views.current_user, name="current_user"),
    path("csrf/", views.csrf_token, name="csrf_token"),

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
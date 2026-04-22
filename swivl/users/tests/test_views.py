from http import HTTPStatus

import pytest
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.models import AnonymousUser
from django.contrib.messages.middleware import MessageMiddleware
from django.contrib.sessions.middleware import SessionMiddleware
from django.http import HttpRequest
from django.http import HttpResponseRedirect
from django.test import RequestFactory
from django.urls import reverse
from django.utils.translation import gettext_lazy as _

from swivl.users.forms import UserAdminChangeForm
from swivl.users.models import User
from swivl.users.tests.factories import UserFactory
from swivl.users.views import UserRedirectView
from swivl.users.views import UserUpdateView
from swivl.users.views import google_login_redirect_view
from swivl.users.views import microsoft_login_redirect_view
from swivl.users.views import user_detail_view

pytestmark = pytest.mark.django_db


class TestUserUpdateView:
    """
    TODO:
        extracting view initialization code as class-scoped fixture
        would be great if only pytest-django supported non-function-scoped
        fixture db access -- this is a work-in-progress for now:
        https://github.com/pytest-dev/pytest-django/pull/258
    """

    def dummy_get_response(self, request: HttpRequest):
        return None

    def test_get_success_url(self, user: User, rf: RequestFactory):
        view = UserUpdateView()
        request = rf.get("/fake-url/")
        request.user = user

        view.request = request
        assert view.get_success_url() == f"/users/{user.username}/"

    def test_get_object(self, user: User, rf: RequestFactory):
        view = UserUpdateView()
        request = rf.get("/fake-url/")
        request.user = user

        view.request = request

        assert view.get_object() == user

    def test_form_valid(self, user: User, rf: RequestFactory):
        view = UserUpdateView()
        request = rf.get("/fake-url/")

        # Add the session/message middleware to the request
        SessionMiddleware(self.dummy_get_response).process_request(request)
        MessageMiddleware(self.dummy_get_response).process_request(request)
        request.user = user

        view.request = request

        # Initialize the form
        form = UserAdminChangeForm()
        form.cleaned_data = {}
        form.instance = user
        view.form_valid(form)

        messages_sent = [m.message for m in messages.get_messages(request)]
        assert messages_sent == [_("Information successfully updated")]


class TestUserRedirectView:
    def test_get_redirect_url(self, user: User, rf: RequestFactory):
        view = UserRedirectView()
        request = rf.get("/fake-url")
        request.user = user

        view.request = request
        assert view.get_redirect_url() == f"/users/{user.username}/"


class TestUserDetailView:
    def test_authenticated(self, user: User, rf: RequestFactory):
        request = rf.get("/fake-url/")
        request.user = UserFactory()
        response = user_detail_view(request, username=user.username)

        assert response.status_code == HTTPStatus.OK

    def test_not_authenticated(self, user: User, rf: RequestFactory):
        request = rf.get("/fake-url/")
        request.user = AnonymousUser()
        response = user_detail_view(request, username=user.username)
        login_url = reverse(settings.LOGIN_URL)

        assert isinstance(response, HttpResponseRedirect)
        assert response.status_code == HTTPStatus.FOUND
        assert response.url == f"{login_url}?next=/fake-url/"


def test_google_login_redirect_view_uses_frontend_next(rf: RequestFactory, settings):
    settings.FRONTEND_BASE_URL = "https://app.amplerank.ai"
    request = rf.get("/auth/google/login/", {"next": "/app"})
    response = google_login_redirect_view(request)
    assert response.status_code == HTTPStatus.FOUND
    assert (
        response.url
        == "/accounts/google/login/?next=https%3A%2F%2Fapp.amplerank.ai%2Fapp"
    )


def test_microsoft_login_redirect_view_uses_frontend_next(rf: RequestFactory, settings):
    settings.FRONTEND_BASE_URL = "https://app.amplerank.ai"
    request = rf.get("/auth/microsoft/login/", {"next": "/app"})
    response = microsoft_login_redirect_view(request)
    assert response.status_code == HTTPStatus.FOUND
    assert (
        response.url
        == "/accounts/microsoft/login/?next=https%3A%2F%2Fapp.amplerank.ai%2Fapp"
    )

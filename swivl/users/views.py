from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.messages.views import SuccessMessageMixin
from django.db.models import QuerySet
from django.http import HttpRequest, HttpResponseRedirect
from django.shortcuts import redirect
from django.urls import reverse
from django.utils.translation import gettext_lazy as _
from django.views.generic import DetailView
from django.views.generic import RedirectView
from django.views.generic import UpdateView
from django.conf import settings
from urllib.parse import urlencode

from swivl.users.models import User


class UserDetailView(LoginRequiredMixin, DetailView):
    model = User
    slug_field = "username"
    slug_url_kwarg = "username"


user_detail_view = UserDetailView.as_view()


class UserUpdateView(LoginRequiredMixin, SuccessMessageMixin, UpdateView):
    model = User
    fields = ["name"]
    success_message = _("Information successfully updated")

    def get_success_url(self) -> str:
        assert self.request.user.is_authenticated  # type guard
        return self.request.user.get_absolute_url()

    def get_object(self, queryset: QuerySet | None = None) -> User:
        assert self.request.user.is_authenticated  # type guard
        return self.request.user


user_update_view = UserUpdateView.as_view()


class UserRedirectView(LoginRequiredMixin, RedirectView):
    permanent = False

    def get_redirect_url(self) -> str:
        return reverse("users:detail", kwargs={"username": self.request.user.username})


user_redirect_view = UserRedirectView.as_view()


def google_login_redirect_view(request: HttpRequest) -> HttpResponseRedirect:
    """
    Entry point used by the frontend Google button.

    Redirects into django-allauth's Google login view and passes along an
    optional `next` parameter so that, after Google completes, the user is
    sent back to the desired frontend path (e.g. `/onboarding`).
    """
    frontend_next = request.GET.get("next") or "/onboarding"
    if not frontend_next.startswith(("http://", "https://")):
        base = getattr(settings, "FRONTEND_BASE_URL", "http://localhost:3000").rstrip("/")
        path = frontend_next if frontend_next.startswith("/") else f"/{frontend_next}"
        frontend_next = f"{base}{path}"

    qs = urlencode({"next": frontend_next})
    return redirect(f"/accounts/google/login/?{qs}")
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
    sent back to the desired frontend path (e.g. `/app`).
    """
    next_url = request.GET.get("next") or "/app"

    if not isinstance(next_url, str) or not next_url.startswith("/"):
        next_url = "/app"

    login_path = "/accounts/google/login/"
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print(f"{login_path}?next={next_url}")
    return redirect(f"{login_path}?next={next_url}")
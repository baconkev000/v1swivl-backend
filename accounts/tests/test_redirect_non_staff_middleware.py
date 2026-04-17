from django.contrib.auth.models import AnonymousUser
from django.http import HttpResponse
from django.test import RequestFactory

from accounts.middleware import RedirectNonStaffApiHostPagesMiddleware


def test_middleware_passes_through_django_admin_for_anonymous(settings):
    settings.ADMIN_URL = "admin/"
    rf = RequestFactory()
    request = rf.get("/admin/login/")
    request.user = AnonymousUser()

    def get_response(_req):
        return HttpResponse("ok")

    resp = RedirectNonStaffApiHostPagesMiddleware(get_response)(request)
    assert resp.status_code == 200
    assert resp.content == b"ok"


def test_middleware_redirects_root_for_anonymous(settings):
    settings.ADMIN_URL = "admin/"
    rf = RequestFactory()
    request = rf.get("/")
    request.user = AnonymousUser()

    def get_response(_req):
        return HttpResponse("ok")

    resp = RedirectNonStaffApiHostPagesMiddleware(get_response)(request)
    assert resp.status_code == 302
    assert "localhost:3000" in resp["Location"] or "/app" in resp["Location"]

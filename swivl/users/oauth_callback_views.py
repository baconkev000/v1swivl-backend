"""OAuth2 provider callbacks with SPA-friendly final navigation (see ``spa_oauth_redirect``)."""

from __future__ import annotations

from allauth.socialaccount.providers.google.views import GoogleOAuth2Adapter
from allauth.socialaccount.providers.microsoft.views import MicrosoftGraphOAuth2Adapter
from allauth.socialaccount.providers.oauth2.views import OAuth2CallbackView

from swivl.users.spa_oauth_redirect import maybe_wrap_redirect_for_spa_history


class SPAHistoryFriendlyOAuth2CallbackView(OAuth2CallbackView):
    def dispatch(self, request, *args, **kwargs):
        response = super().dispatch(request, *args, **kwargs)
        return maybe_wrap_redirect_for_spa_history(response)


google_oauth_callback_view = SPAHistoryFriendlyOAuth2CallbackView.adapter_view(GoogleOAuth2Adapter)
microsoft_oauth_callback_view = SPAHistoryFriendlyOAuth2CallbackView.adapter_view(
    MicrosoftGraphOAuth2Adapter
)

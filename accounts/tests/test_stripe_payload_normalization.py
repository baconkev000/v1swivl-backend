import pytest

from accounts.stripe_billing import normalize_stripe_payload


class _ObjToDictRecursive:
    def __init__(self, payload):
        self._payload = payload

    def to_dict_recursive(self):
        return self._payload


class _ObjPrivateToDictRecursive:
    def __init__(self, payload):
        self._payload = payload

    def _to_dict_recursive(self):
        return self._payload


class _MappingLike:
    def __init__(self, payload):
        self._payload = payload

    def items(self):
        return self._payload.items()


class _OpaqueObject:
    pass


def test_normalize_stripe_payload_handles_to_dict_recursive():
    raw = _ObjToDictRecursive({"id": "cus_123", "invoice_settings": {"default_payment_method": None}})
    out = normalize_stripe_payload(raw)
    assert isinstance(out, dict)
    assert out["id"] == "cus_123"
    assert out["invoice_settings"]["default_payment_method"] is None


def test_normalize_stripe_payload_handles_private_to_dict_recursive():
    raw = _ObjPrivateToDictRecursive({"id": "sub_123", "status": "active"})
    out = normalize_stripe_payload(raw)
    assert isinstance(out, dict)
    assert out["id"] == "sub_123"
    assert out["status"] == "active"


def test_normalize_stripe_payload_recurses_nested_objects_lists_and_mappings():
    raw = _ObjToDictRecursive(
        {
            "data": [
                _ObjPrivateToDictRecursive(
                    {
                        "id": "in_123",
                        "payment_intent": _MappingLike({"id": "pi_123", "payment_method": None}),
                    }
                )
            ]
        }
    )
    out = normalize_stripe_payload(raw)
    assert isinstance(out, dict)
    assert isinstance(out["data"], list)
    assert out["data"][0]["id"] == "in_123"
    assert out["data"][0]["payment_intent"]["id"] == "pi_123"


def test_normalize_stripe_payload_non_convertible_object_fallback_safe():
    raw = _OpaqueObject()
    out = normalize_stripe_payload(raw)
    assert out is raw

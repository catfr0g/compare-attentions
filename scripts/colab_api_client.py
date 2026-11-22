"""Bounded JSON transport and explicit safe projections for Colab GETs."""

import http.client
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid

BASE = "https://colaboratory.googleapis.com/v1beta/"
SPEC = {"variant": "VARIANT_GPU", "accelerator": "T4", "shape": "SHAPE_STANDARD"}
REASONS = {
    "SERVICE_DISABLED",
    "ACCESS_TOKEN_SCOPE_INSUFFICIENT",
    "IAM_PERMISSION_DENIED",
    "API_KEY_INVALID",
    "RATE_LIMIT_EXCEEDED",
    "QUOTA_EXCEEDED",
}
LIMIT = 2 * 1024 * 1024


class ApiError(Exception):
    def __init__(self, error, **fields):
        self.payload = {"error": error, **fields}
        super().__init__(json.dumps(self.payload))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request_json(url, *, headers=None, data=None, opener=None, method=None):
    request = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    open_request = opener or urllib.request.build_opener(NoRedirect()).open
    try:
        with open_request(request, timeout=20) as response:
            raw = response.read(LIMIT + 1)
        if len(raw) > LIMIT:
            raise ApiError("RESPONSE_TOO_LARGE")
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ApiError("INVALID_RESPONSE")
        return result
    except urllib.error.HTTPError as exc:
        fields = {"status": exc.code}
        try:
            body = json.loads(exc.read(LIMIT))
            for detail in body.get("error", {}).get("details", []):
                if (
                    detail.get("@type") == "type.googleapis.com/google.rpc.ErrorInfo"
                    and detail.get("reason") in REASONS
                ):
                    fields["reason"] = detail["reason"]
        except (ValueError, AttributeError, TypeError, OSError, http.client.HTTPException):
            pass
        finally:
            exc.close()
        raise ApiError("HTTP_ERROR", **fields) from None
    except http.client.HTTPException:
        raise ApiError("TRANSPORT_ERROR") from None
    except (OSError, ValueError, TypeError) as exc:
        raise ApiError(
            "TRANSPORT_ERROR" if isinstance(exc, OSError) else "INVALID_RESPONSE"
        ) from None


def get(url, token):
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "colaboratory.googleapis.com"
        or parsed.query
        or parsed.fragment
        or not re.fullmatch(r"/v1beta/(runtimespecs|runtimes(?:/[a-zA-Z0-9_-]+)?)", parsed.path)
    ):
        raise ApiError("INVALID_ENDPOINT")
    return request_json(url, headers={"Authorization": "Bearer " + token})


def validate_runtime_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?", value):
        raise ApiError("INVALID_RUNTIME_ID")
    return value


def validate_request_id(value):
    try:
        parsed = uuid.UUID(value)
    except (ValueError, TypeError, AttributeError):
        raise ApiError("INVALID_REQUEST_ID") from None
    if parsed.version != 4 or str(parsed) != value:
        raise ApiError("INVALID_REQUEST_ID")
    return value


def operation_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"operations/[A-Za-z0-9_-]+", value):
        raise ApiError("INVALID_OPERATION_NAME")
    return value


def operation(name, token):
    return request_json(
        "https://colaboratory.googleapis.com/v1/" + operation_name(name),
        headers={"Authorization": "Bearer " + token},
    )


def manage(method, name, token, *, request_id=None, runtime_id=None):
    headers = {"Authorization": "Bearer " + token}
    if method == "POST" and name == "runtimes":
        # Validate literal caller values; never normalize, repair or regenerate them.
        query = urllib.parse.urlencode(
            {
                "requestId": validate_request_id(request_id),
                "runtimeId": validate_runtime_id(runtime_id),
            }
        )
        return request_json(
            BASE + name + "?" + query,
            headers={**headers, "Content-Type": "application/json"},
            data=json.dumps({"runtimeSpec": SPEC}).encode(),
            method="POST",
        )
    if method == "DELETE" and isinstance(name, str) and name.startswith("runtimes/"):
        validate_runtime_id(name.removeprefix("runtimes/"))
        if request_id is not None or runtime_id is not None:
            raise ApiError("INVALID_ARGUMENT")
        return request_json(BASE + name, headers=headers, method="DELETE")
    raise ApiError("INVALID_ENDPOINT")


def key(raw):
    if not isinstance(raw, dict):
        raise ApiError("INVALID_RESPONSE")
    return {
        name: raw[name]
        for name in ("variant", "accelerator", "shape")
        if isinstance(raw.get(name), str)
    }


def specs(raw, *, t4_only=False):
    rows = raw.get("runtimeSpecs", [])
    if not isinstance(rows, list):
        raise ApiError("INVALID_RESPONSE")
    result = []
    for row in rows:
        if not isinstance(row, dict):
            raise ApiError("INVALID_RESPONSE")
        selected = key(row.get("key", {}))
        eligible = row.get("eligible") is True
        if not t4_only or (
            eligible
            and selected.get("variant") == "VARIANT_GPU"
            and selected.get("accelerator") == "T4"
            and selected.get("shape") in ("SHAPE_STANDARD", "SHAPE_HIGHMEM")
        ):
            result.append({"key": selected, "eligible": eligible})
    return result


def runtime(raw):
    if not isinstance(raw, dict):
        raise ApiError("INVALID_RESPONSE")
    result = {name: raw[name] for name in ("name", "version") if isinstance(raw.get(name), str)}
    result["runtimeSpec"] = key(raw.get("runtimeSpec", {}))
    return result

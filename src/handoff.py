
from __future__ import annotations

import ipaddress

import requests
import streamlit as st

def _cfg(key: str, default=None):
    return st.secrets.get(key, default)

# --------------------------------------------------------------------------- #
# One-time code redemption (server-to-server)
# --------------------------------------------------------------------------- #

def redeem_code(code: str) -> dict | None:
    """Exchange a one-time code for user claims.

    Runs in the Streamlit backend (never the browser). Returns a minimal claims
    dict on success, or None on any failure — failures are intentionally opaque.
    """
    try:
        resp = requests.post(
            _cfg("PHP_REDEEM_URL"),
            json={"code": code},
            headers={"X-Handoff-Secret": _cfg("HANDOFF_SHARED_SECRET")},
            timeout=5,
            verify=bool(_cfg("VERIFY_TLS", True)),
        )
    except requests.RequestException:
        return None

    if resp.status_code != 200:
        return None
    try:
        claims = resp.json()
    except ValueError:
        return None

    if not isinstance(claims, dict) or "sub" not in claims:
        return None
    return claims

# --------------------------------------------------------------------------- #
# Trusted-network bypass
# --------------------------------------------------------------------------- #

def _ip_in_cidrs(ip: str, cidrs) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    for c in cidrs or []:
        try:
            if addr in ipaddress.ip_network(str(c), strict=False):
                return True
        except ValueError:
            continue
    return False

def _tornado_request():
    """Best-effort handle on the Tornado request behind the current session.

    Uses Streamlit internals, so it can change between versions — always wrapped.
    """
    try:
        from streamlit.runtime import get_instance
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx()
        if ctx is None:
            return None

        instance = get_instance()
        client = instance.get_client(ctx.session_id)

        # Streamlit 1.41+: StarletteSessionClient doesn't have .request directly
        # Try _client_context first (newer versions), then fall back to others
        req = getattr(client, "request", None)

        if req is None:
            # Try _client_context (Streamlit 1.41+)
            client_context = getattr(client, "_client_context", None)
            if client_context:
                req = getattr(client_context, "request", None)

        print(f"[DEBUG] Found request: {req}")
        return req

    except Exception as e:
        print(f"[DEBUG] Exception in _tornado_request: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None



def _trusted_network() -> bool:
    """Whether this request may skip the login, based on network trust.

    SECURITY: in the recommended "proxy_header" mode the decision is anchored at
    the reverse proxy, which sets X-Trusted-Network unconditionally. We never
    trust a client-supplied X-Forwarded-For to make this call — it is spoofable.
    """
    if not bool(_cfg("TRUSTED_NETWORK_ENABLED", False)):
        return False

    req = _tornado_request()
    print("TRUSTED NETWORK CHECK:", req)
    if req is None:
        return False

    mode = _cfg("TRUSTED_NETWORK_MODE", "proxy_header")

    if mode == "proxy_header":
        # Value is trustworthy because the proxy overwrites any client-sent one.
        try:
            return req.headers.get("X-Trusted-Network", "0") == "1"
        except Exception:
            return False

    if mode == "remote_ip":
        # Correct ONLY if Streamlit is NOT behind a proxy: otherwise remote_ip
        # is the proxy's address, not the real client's, and this is meaningless.
        ip = getattr(req, "remote_ip", None)
        return bool(ip) and _ip_in_cidrs(ip, _cfg("TRUSTED_NETWORK_CIDRS", []))

    return False

def _trusted_network_user() -> dict:
    """A configured pseudo-identity for trusted-network access.

    NOTE: an IP is not a person. Everyone on the trusted network shares this one
    identity, so you lose per-user attribution. Give it a role that grants
    exactly what an anonymous internal visitor should be allowed to do.
    """
    return {
        "sub": _cfg("TRUSTED_NETWORK_SUB", "internal"),
        "display_name": _cfg("TRUSTED_NETWORK_DISPLAY_NAME", "Internes Netz"),
        "role": _cfg("TRUSTED_NETWORK_ROLE", "internal"),
        "via": "trusted_network",
        "group": "test"
    }


# --------------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------------- #

def require_login() -> dict:
    """Gate the app. Returns the user's claims, or halts the script rendering."""
    user = st.session_state.get("user")
    if user:
        return user

    # DEBUG MODE: Skip login for local development
    if _cfg("DEBUG_SKIP_LOGIN", False):
        st.session_state["user"] = {
            "sub": "debug-user",
            "display_name": "Debug User",
            "role": "admin",
            "via": "debug_mode",
            "group": "dev"
        }
        return st.session_state["user"]

    # Trusted-network bypass runs before the code flow.
    if _trusted_network():
        st.session_state["user"] = _trusted_network_user()
        return st.session_state["user"]

    code = st.query_params.get("code")
    if code:
        claims = redeem_code(code)
        if claims:
            st.session_state["user"] = claims
            st.query_params.clear()  # drop ?code from the browser URL
            st.rerun()
        st.error("Anmeldung fehlgeschlagen oder Link abgelaufen. Bitte erneut anmelden.")
        st.link_button("Zur Anmeldung", _cfg("PHP_LOGIN_URL"))
        st.stop()

    st.title("Anmeldung erforderlich")
    st.write("Bitte melden Sie sich über Ihr Konto an, um Zugang zu erhalten.")
    st.link_button("Zur Anmeldung", _cfg("PHP_LOGIN_URL"))
    st.stop()
    
def logout() -> None:
    st.session_state.pop("user", None)
    st.rerun()
# Proxy Configuration Request for Multiple Streamlit Apps

## Current Situation

We have two Streamlit apps running:
1. **Main app** (`bbs3.py`) on port **8501** - accessible at `https://chatbots.daisec.eu/`
2. **R+S Auskunft** (`rsev.py`) on port **8502** - should be accessible at `https://chatbots.daisec.eu/rsev/`

## Problem

Currently, the proxy configuration routes ALL traffic (`/`) to port 8501:

```apache
<Location "/">
    ProxyPass http://10.10.150.58:8501/
    ProxyPassReverse http://10.10.150.58:8501/
</Location>
```

This means requests to `/rsev/` are being forwarded to port 8501 instead of port 8502, causing a "Page not found" error.

## Required Proxy Configuration

To support multiple apps, the proxy needs to route `/rsev/` (and its subpaths) to port 8502:

```apache
# Route /rsev/ to the RSEV app on port 8502
<LocationMatch "^/rsev(/.*)?$">
    ProxyPass http://10.10.150.58:8502$1
    ProxyPassReverse http://10.10.150.58:8502$1
</LocationMatch>

# WebSockets for /rsev/
<LocationMatch "^/rsev/_stcore(/.*)?$">
    ProxyPass ws://10.10.150.58:8502/_stcore$1
    ProxyPassReverse ws://10.10.150.58:8502/_stcore$1
</LocationMatch>

# Default route for main app (must come AFTER specific routes)
<Location "/">
    ProxyPass http://10.10.150.58:8501/
    ProxyPassReverse http://10.10.150.58:8501/
</Location>

# WebSockets for main app
<Location "/_stcore/">
    ProxyPass ws://10.10.150.58:8501/_stcore/
    ProxyPassReverse ws://10.10.150.58:8501/_stcore/
</Location>
```

**Important**: The specific routes (`/rsev/`) must be defined BEFORE the catch-all route (`/`) so Apache matches them first.

## Streamlit Configuration

The RSEV app is already configured with:
- Port: 8502
- `baseUrlPath = "rsev"` in `/home/dren.fazlija/rolladen/.streamlit/config.toml`
- Service: `streamlit-rsev.service` (running and enabled)

## Testing

After the proxy configuration is updated:
1. Access `https://chatbots.daisec.eu/rsev/` (with trailing slash)
2. The app should load correctly
3. WebSocket connections should work for the app

## Future Apps

To add more apps in the future:
1. Run app on a unique port (e.g., 8503, 8504, etc.)
2. Set `baseUrlPath` in the app's config (e.g., `baseUrlPath = "appname"`)
3. Add corresponding `<LocationMatch>` blocks in the proxy config

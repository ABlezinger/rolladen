# Troubleshooting 404 Error Checklist

## Step-by-Step Diagnosis

### 1. Verify Streamlit is Running
```bash
# Check if port 8501 is listening
netstat -tuln | grep 8501
# OR
ss -tuln | grep 8501

# Test direct connection
curl http://localhost:8501/bbs3/
```

**Expected:** Should return HTML content, not connection refused

### 2. Verify Apache Configuration Was Updated
```bash
# Check current config
sudo cat /etc/apache2/sites-available/bbs3.conf
```

**Look for:**
- ✅ `ProxyPass /bbs3 http://localhost:8501/bbs3` (NOT https)
- ✅ `ProxyPassReverse /bbs3 http://localhost:8501/bbs3` (NOT https)
- ✅ WebSocket rewrite uses `ws://localhost:8501` (NOT wss://)

### 3. Verify Apache Site is Enabled
```bash
# Check if site is enabled
ls -la /etc/apache2/sites-enabled/ | grep bbs3

# If not enabled:
sudo a2ensite bbs3
sudo systemctl reload apache2
```

### 4. Verify Required Apache Modules
```bash
# Check enabled modules
apache2ctl -M | grep -E "proxy|rewrite|ssl"

# Enable missing modules:
sudo a2enmod proxy
sudo a2enmod proxy_http
sudo a2enmod proxy_wstunnel
sudo a2enmod rewrite
sudo a2enmod ssl
sudo systemctl reload apache2
```

### 5. Check Apache Error Logs
```bash
# View recent errors
sudo tail -50 /var/log/apache2/error.log

# Common errors to look for:
# - "proxy: error reading status line from remote server"
# - "Connection refused"
# - "Name or service not known"
```

### 6. Check Apache Access Logs
```bash
# See what requests are coming in
sudo tail -50 /var/log/apache2/access.log | grep bbs3
```

### 7. Test Apache Configuration Syntax
```bash
sudo apache2ctl configtest
```

**Expected:** "Syntax OK"

## Common Issues and Fixes

### Issue 1: Apache config still pointing to HTTPS
**Symptom:** 404 or connection errors
**Fix:** Update config to use `http://localhost:8501` instead of `https://chatbots.daisec.eu:8501`

### Issue 2: Site not enabled
**Symptom:** Apache doesn't serve the site at all
**Fix:** `sudo a2ensite bbs3 && sudo systemctl reload apache2`

### Issue 3: Missing trailing slash
**Symptom:** Works with `/bbs3/` but not `/bbs3`
**Fix:** Ensure ProxyPass includes trailing slash: `/bbs3/` → `/bbs3/`

### Issue 4: Streamlit not running
**Symptom:** 502 Bad Gateway or connection refused
**Fix:** Start Streamlit first, then reload Apache

### Issue 5: Wrong baseUrlPath
**Symptom:** Streamlit serves at root but Apache expects /bbs3
**Fix:** Ensure Streamlit command includes `--server.baseUrlPath bbs3`

## Quick Test Commands

```bash
# 1. Run diagnostic script
./diagnose_404.sh

# 2. Test Streamlit directly
curl http://localhost:8501/bbs3/

# 3. Test Apache locally
curl -k https://localhost/bbs3/

# 4. Check Apache status
sudo systemctl status apache2
```

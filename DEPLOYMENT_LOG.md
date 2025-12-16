# Housing App Deployment Log - Part #1

## Student: Shreyas Aravind
## Date: December 16, 2024

---

## Part #1: Infrastructure Setup - COMPLETE ✅

### Local Deployment
- Forked repository from mkzia/housing_app_fall25
- Cloned to local Mac
- Ran `docker-compose up -d`
- Verified at http://localhost:8501 and http://localhost:8000

### Cloud Deployment (DigitalOcean)
- Created Droplet: Ubuntu 24.04, $6/month, 1GB RAM
- IP Address: **138.197.37.164**
- Installed Docker and Docker Compose
- Cloned repository to droplet
- Fixed docker-compose.yml (changed expose to ports)
- Configured firewall (UFW)
- Started application with docker-compose

### Live URLs
- Streamlit: http://138.197.37.164:8501 ✅
- API: http://138.197.37.164:8000 ✅
- API Docs: http://138.197.37.164:8000/docs ✅

### Key Commands Used
```
ssh root@138.197.37.164
docker-compose up -d
docker-compose ps
ufw allow 8000/tcp
ufw allow 8501/tcp
```

### Status
Part #1 Complete: 15/15 points ✅

---

## Next: Part #2 - Machine Learning Work

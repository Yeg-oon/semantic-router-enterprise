# Enterprise Semantic Router - Live Demo

> **Production-Grade Intent Router with Circuit Breakers & Metrics**

## Live Demo
**Try it now**: https://semantic-router-enterprise.onrender.com

## Features
- **Circuit Breaker**: Automatic failover protection
- **Prometheus Metrics**: Real-time monitoring at `/metrics`
- **Rate Limiting**: 100 requests/minute per IP
- **Structured Logging**: JSON format for enterprise monitoring
- **Health Checks**: `/health` endpoint
- **X-Router-Path Header**: For audit trails

## Quick Test

### Health Check
```bash
curl https://semantic-router-enterprise.onrender.com/health
```

### Test Routing
```bash
curl -X POST https://semantic-router-enterprise.onrender.com/v1/route \
  -H "Content-Type: application/json" \
  -d '{"message": "Write a Python script to sort a list"}'
```

### View Metrics
```bash
curl https://semantic-router-enterprise.onrender.com/metrics
```

## Enterprise Edition
This is the **Enterprise version** with all production features enabled.

For the **Lite (free) version**, visit:
https://github.com/Yeg-oon/enterprise-semantic-router

## Architecture
```
Request Classifier (GPT-4o-mini)
    |
    v
Simple Support  ->  Fast Model (GPT-3.5-turbo)
Complex Task    ->  Smart Model (GPT-4o)
```

## Cost Savings
- **60% reduction** in LLM costs
- **Automatic routing** to optimal models
- **Enterprise reliability** with circuit breakers

## Support
- **Enterprise Support**: Direct access to developers
- **Documentation**: Complete API reference
- **Monitoring**: Built-in metrics and health checks

---
*Powered by Enterprise Semantic Router v2.0*

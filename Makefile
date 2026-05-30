# ============================================================
# Makefile — Cycling Training App
# ============================================================

APP_NAME    := cycling-training-app
REGISTRY    ?= ghcr.io/kvarnberg-labs/cycling-training-app
TAG         ?= latest
NAMESPACE   ?= cycling-training-app
K8S_DIR     := k8s

# ── Docker ──────────────────────────────────────────────────

.PHONY: build push

build:
	docker build -t $(REGISTRY):$(TAG) .

push:
	docker push $(REGISTRY):$(TAG)

# ── Kubernetes ──────────────────────────────────────────────

.PHONY: deploy logs restart stop

deploy:
	kubectl apply -f $(K8S_DIR)/configmap.yaml
	kubectl apply -f $(K8S_DIR)/secret.yaml
	kubectl apply -f $(K8S_DIR)/deployment.yaml
	kubectl apply -f $(K8S_DIR)/service.yaml
	kubectl apply -f $(K8S_DIR)/ingress.yaml
	@echo "✓ Deployed to namespace: $(NAMESPACE)"

logs:
	kubectl logs -n $(NAMESPACE) -l app=$(APP_NAME) --tail=100 -f

restart:
	kubectl rollout restart -n $(NAMESPACE) deployment/$(APP_NAME)
	@echo "✓ Restarted deployment"

stop:
	kubectl scale deployment/$(APP_NAME) -n $(NAMESPACE) --replicas=0
	@echo "✓ Scaled to 0 replicas"

# ── Help ────────────────────────────────────────────────────

.PHONY: help

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Docker targets:"
	@echo "  build        Build Docker image: $(REGISTRY):$(TAG)"
	@echo "  push         Push Docker image to registry"
	@echo ""
	@echo "Kubernetes targets:"
	@echo "  deploy       Deploy all manifests to cluster"
	@echo "  logs         Tail logs from running pods"
	@echo "  restart      Rollout restart the deployment"
	@echo "  stop         Scale deployment to 0 replicas"

# Amadeus Transfer Assistant

A conversational FastAPI service to search, book, and cancel Amadeus car transfers.

## Requirements
- Python 3.12 (for local runs) or Docker
- Amadeus API credentials: `AMADEUS_API_KEY`, `AMADEUS_API_SECRET`

## Quick Start (Docker)

```bash
# Set credentials in your shell
export AMADEUS_API_KEY=... 
export AMADEUS_API_SECRET=...

# Build and run
docker compose up --build
# App on http://localhost:8000, DB is Postgres in the compose stack
```

Health check:
```bash
curl http://localhost:8000/health
```

Chat examples:
```bash
# Start a new session
curl -s -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"search from CDG to ORY at 2025-01-01 10:00 for 2 passengers"}' | jq

# Use returned session_id to continue
curl -s -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"<SESSION>","message":"book offerId: <OFFER_ID> {\n  \"passengers\":[{\"firstName\":\"John\",\"lastName\":\"Doe\",\"title\":\"MR\",\"contacts\":{\"phoneNumber\":\"+33123456789\",\"email\":\"john@example.com\"}}],\n  \"payment\":{\"methodOfPayment\":\"CREDIT_CARD\",\"creditCard\":{\"number\":\"4111111111111111\",\"holderName\":\"JOHN DOE\",\"vendorCode\":\"VI\",\"expiryDate\":\"1025\",\"cvv\":\"123\"}}\n}' }' | jq

# Cancel
curl -s -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"<SESSION>","message":"cancel orderId: <ORDER_ID> confirmNbr: <CONFIRM>"}' | jq
```

## Local Run (no Docker)

```bash
pip install -e .
uvicorn ai:app --reload --host 0.0.0.0 --port 8000
```

Set `DATABASE_URL` to use PostgreSQL if desired, otherwise SQLite is used by default.

## Notes
- The assistant stores conversation history and last results in the database.
- It uses basic intent detection and will ask for missing fields.
- Provide JSON blocks to pass complex parameters easily.

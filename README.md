# 🚗 Amadeus Transfer Agent

An AI-powered travel assistant agent that enables users to **search**, **book**, and **cancel** ground transfers using the [Amadeus Transfer API](https://developers.amadeus.com/). Built with [LangGraph](https://langchain-ai.github.io/langgraph/) and deployed as a decentralized agent on the [Fetch.ai uAgents](https://fetch.ai/docs/guides/agents/getting-started/installing-uagent) network.

## ✨ Features

- **🔍 Transfer Search** – Find available ground transfers between airports and addresses worldwide
- **📝 Transfer Booking** – Book selected transfer offers with passenger details
- **❌ Transfer Cancellation** – Cancel existing bookings with order ID and confirmation number
- **🤖 AI-Powered Chat** – Natural language interface powered by GPT-4o-mini
- **💾 Persistent Memory** – Conversation history stored in PostgreSQL using LangGraph checkpointing
- **🌐 Decentralized Agent** – Runs on the Fetch.ai testnet with mailbox support

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Fetch.ai      │────▶│   LangGraph     │────▶│   Amadeus API   │
│   uAgent        │     │   AI Agent      │     │   (Transfers)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │
         │                       ▼
         │              ┌─────────────────┐
         └─────────────▶│   PostgreSQL    │
                        │   (Checkpoints) │
                        └─────────────────┘
```

## 📁 Project Structure

```
amadeus-agent/
├── app/
│   ├── main.py          # Application entrypoint
│   ├── agent.py         # Fetch.ai uAgent with chat protocol
│   ├── ai.py            # LangGraph AI agent with tools
│   ├── api.py           # Amadeus Transfer API client (async, typed)
│   └── prompt.md        # System prompt for the AI assistant
├── swagger_specification/
│   └── *.json           # Amadeus API specifications (v1.11)
├── typings/             # Type stubs for uagents
├── docker-compose.yml   # Docker services configuration
├── Dockerfile           # Application container
└── pyproject.toml       # Project dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- Docker & Docker Compose
- [Amadeus API credentials](https://developers.amadeus.com/)
- [OpenAI API key](https://platform.openai.com/)

### Environment Variables

Create a `.env` file in the project root:

```env
# Amadeus API
AMADEUS_API_KEY=your_amadeus_api_key
AMADEUS_API_SECRET=your_amadeus_api_secret

# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Agent Configuration
AGENT_SEED=your-unique-agent-seed
AGENT_PORT=8000

# PostgreSQL
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_HOST=database
POSTGRES_PORT=5432
POSTGRES_DB=amadeus_agent
```

### Running with Docker (Recommended)

```bash
# Build and start all services
docker-compose up --build

# Run in detached mode
docker-compose up -d --build
```

### Running Locally

```bash
# Install dependencies with uv
uv sync

# Run the agent
uv run python -m app.main
```

## 🛠️ Available Tools

The AI agent has access to three main tools:

| Tool | Description | Required Parameters |
|------|-------------|---------------------|
| `search_transfers` | Search for available ground transfers | `start`, `end`, `startDateTime`, `passengers` |
| `book_transfer` | Book a selected transfer offer | `offerId`, `passengers`, `payment` |
| `cancel_transfer` | Cancel an existing booking | `orderId`, `confirmationNumber` |

### Location Types

- **Airport**: Use IATA codes (e.g., `CDG`, `JFK`, `LHR`)
- **Address**: Provide full address with country code and geo coordinates

## 💬 Example Conversation

```
User: I need a transfer from CDG airport to the Eiffel Tower tomorrow at 2pm

Agent: I found 5 transfer options for you:

• Offer #ABC123 - €45.00 EUR
  🚗 Mercedes E-Class (Sedan)
  📍 Provider: Paris Transfers
  
• Offer #DEF456 - €65.00 EUR
  🚐 Mercedes V-Class (Van)
  📍 Provider: Elite Chauffeurs

Would you like to book one of these?

User: I'll take the first one. My name is John Smith.

Agent: To complete the booking, I need:
- Title (Mr/Mrs/Ms/Miss)
- Phone number with country code
- Email address

User: Mr, +33123456789, john@email.com

Agent: ✅ Booking successful!
- Order ID: ORD-789012
- Confirmation Number: CONF-345678
```

## 🧪 Development

```bash
# Install dev dependencies
uv sync --dev

# Format code
uv run black .

# Lint
uv run ruff check .

# Type checking
uv run mypy .

# Run tests
uv run pytest
```

## 📚 API Reference

This agent uses the following Amadeus Transfer APIs (v1.11):

- **Transfer Search** – Find available transfer offers
- **Transfer Booking** – Book a transfer offer
- **Transfer Management** – Manage and cancel bookings

See the [swagger_specification](swagger_specification/) directory for detailed API schemas.

## 🔗 Resources

- [Amadeus for Developers](https://developers.amadeus.com/)
- [Fetch.ai uAgents Documentation](https://fetch.ai/docs/guides/agents/getting-started/installing-uagent)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain OpenAI](https://python.langchain.com/docs/integrations/llms/openai/)

## 📄 License

MIT License – see [LICENSE](LICENSE) for details.

---

Built with ❤️ using LangGraph, Fetch.ai uAgents, and the Amadeus API

# ETH Global Fetch.ai uAgent - AI Innovation Agent

🚀 **ETH Global Hackathon Submission** - A comprehensive AI agent built on Fetch.ai's uAgents framework that meets all qualification requirements and demonstrates innovative AI capabilities.

## 🎯 Project Overview

This uAgent demonstrates the power of Fetch.ai's technology stack by creating an intelligent AI agent that can be discovered and interacted with through ASI:One at https://asi1.ai. The agent showcases multi-modal AI capabilities, intelligent task routing, and seamless integration with the Fetch.ai ecosystem.

### ✅ ETH Global Requirements Met

1. **✅ Created uAgent** - Built using Fetch.ai's uAgents framework
2. **✅ Hosted on Agentverse** - Deployed to Agentverse.ai with mailbox feature enabled
3. **✅ ASI:One Discoverable** - Agent is discoverable through https://asi1.ai
4. **✅ Chat Protocol Implementation** - Implements Agent Chat Protocol for ASI:One compatibility
5. **✅ Public GitHub Repository** - Complete documentation and open-source code

## 🌟 Key Features

### 🤖 Multi-Modal AI Capabilities
- **Intelligent Query Routing** - Automatically routes queries to appropriate handlers
- **Market Analysis** - Real-time cryptocurrency and DeFi market insights
- **DeFi Recommendations** - Protocol analysis and yield farming strategies
- **NFT Analytics** - Collection analysis and rarity insights
- **Smart Contract Guidance** - Development best practices and security recommendations
- **Web Search Integration** - Real-time information retrieval via Tavily

### 🔗 Fetch.ai Integration
- **Agent Chat Protocol** - Full compatibility with ASI:One
- **Mailbox Feature** - Seamless Agentverse integration
- **LangGraph Integration** - Advanced AI orchestration capabilities
- **LangChain Tools** - Web search and external API integration

### 📊 Analytics & Monitoring
- **Request Tracking** - Comprehensive request statistics
- **Performance Monitoring** - Response time and success rate tracking
- **Error Handling** - Robust error management and recovery
- **Session Management** - Persistent conversation context

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ASI:One       │    │   Agentverse    │    │   uAgent        │
│   (Discovery)   │◄──►│   (Hosting)     │◄──►│   (Core Logic)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Mailbox       │    │   LangGraph     │
                       │   (Messaging)   │    │   (AI Engine)   │
                       └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │   Tavily        │
                                              │   (Web Search)  │
                                              └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Fetch.ai Agentverse API Key
- OpenAI API Key
- Tavily API Key (optional but recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd 42agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

4. **Set required environment variables**
   ```bash
   export AGENTVERSE_API_KEY="your_agentverse_api_key"
   export OPENAI_API_KEY="your_openai_api_key"
   export TAVILY_API_KEY="your_tavily_api_key"  # Optional
   ```

### Local Development

1. **Run the agent locally**
   ```bash
   python eth_global_agent.py
   ```

2. **Test with curl**
   ```bash
   curl http://localhost:8000/test
   ```

### Deployment to Agentverse

1. **Deploy to Agentverse**
   ```bash
   python deploy_to_agentverse.py
   ```

2. **Verify deployment**
   - Visit https://asi1.ai
   - Search for "eth_global_innovator_agent"
   - Test the agent with various queries

## 📋 Usage Examples

### Market Analysis
```
User: "Analyze the current market trends for Ethereum"
Agent: Provides comprehensive market analysis including price trends, 
       risk assessment, and investment recommendations
```

### DeFi Recommendations
```
User: "What are the best DeFi protocols for yield farming?"
Agent: Analyzes current DeFi landscape, provides protocol recommendations,
       and includes risk assessments and APY comparisons
```

### NFT Analytics
```
User: "Analyze the Bored Ape Yacht Club collection"
Agent: Provides collection insights, rarity analysis, floor price trends,
       and market sentiment
```

### Smart Contract Guidance
```
User: "How can I optimize my smart contract for gas efficiency?"
Agent: Provides gas optimization strategies, security best practices,
       and code examples
```

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AGENTVERSE_API_KEY` | Yes | Fetch.ai Agentverse API key |
| `OPENAI_API_KEY` | Yes | OpenAI API key for LLM capabilities |
| `TAVILY_API_KEY` | No | Tavily API key for web search |
| `AGENT_NAME` | No | Custom agent name (default: eth_global_innovator_agent) |
| `AGENT_PORT` | No | Local port (default: 8000) |

### Agent Configuration

The agent can be customized by modifying the following parameters in `eth_global_agent.py`:

```python
# Agent configuration
agent = Agent(
    name=AGENT_NAME,
    seed=AGENT_SEED,
    port=AGENT_PORT,
    endpoint=[f"http://localhost:{AGENT_PORT}/submit"],
    mailbox=True  # Enable Agentverse integration
)
```

## 🧪 Testing

### Local Testing

1. **Start the agent**
   ```bash
   python eth_global_agent.py
   ```

2. **Test REST endpoints**
   ```bash
   curl http://localhost:8000/test
   curl http://localhost:8000/health
   curl http://localhost:8000/info
   ```

3. **Test chat protocol**
   ```bash
   # Use the provided test client
   python test_client.py
   ```

### ASI:One Testing

1. **Deploy to Agentverse**
   ```bash
   python deploy_to_agentverse.py
   ```

2. **Visit ASI:One**
   - Go to https://asi1.ai
   - Search for your agent
   - Test with various queries

## 📊 Monitoring & Analytics

The agent includes comprehensive monitoring capabilities:

- **Request Statistics** - Total requests, success rate, response times
- **Error Tracking** - Detailed error logging and recovery
- **Performance Metrics** - Response time analysis and optimization
- **Usage Analytics** - Query patterns and user interaction insights

### Accessing Metrics

```python
# Get agent statistics
total_requests = ctx.storage.get("total_requests")
successful_requests = ctx.storage.get("successful_requests")
startup_time = ctx.storage.get("startup_time")
```

## 🔒 Security Features

- **Input Validation** - Comprehensive input sanitization
- **Rate Limiting** - Built-in request throttling
- **Error Handling** - Secure error responses
- **Session Management** - Secure session handling
- **API Key Protection** - Environment variable security

## 🚀 Deployment Options

### 1. Local Development
```bash
python eth_global_agent.py
```

### 2. Agentverse Deployment
```bash
python deploy_to_agentverse.py
```

### 3. Docker Deployment
```bash
docker build -t eth-global-agent .
docker run -p 8000:8000 eth-global-agent
```

## 📈 Performance Optimization

### Best Practices

1. **Async Operations** - All operations are async for optimal performance
2. **Connection Pooling** - Reuse connections to external services
3. **Caching** - Implement caching for frequently accessed data
4. **Error Recovery** - Robust error handling and recovery mechanisms
5. **Resource Management** - Proper cleanup of resources

### Monitoring

- **Response Times** - Track and optimize response times
- **Success Rates** - Monitor and improve success rates
- **Resource Usage** - Monitor CPU and memory usage
- **Error Rates** - Track and resolve error patterns

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Fetch.ai Team** - For the amazing uAgents framework
- **ETH Global** - For the hackathon opportunity
- **ASI Alliance** - For the innovative AI ecosystem
- **OpenAI** - For the powerful LLM capabilities
- **Tavily** - For web search integration

## 📞 Support

- **Documentation**: [Fetch.ai Innovation Lab](https://innovationlab.fetch.ai/)
- **Community**: [Fetch.ai Discord](https://discord.gg/fetchai)
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)

## 🎉 ETH Global Submission

This project demonstrates:

- **Innovation** - Multi-modal AI capabilities with intelligent routing
- **Technical Excellence** - Robust architecture with comprehensive error handling
- **User Experience** - Seamless integration with ASI:One for discovery
- **Scalability** - Designed for production deployment and scaling
- **Documentation** - Comprehensive documentation and examples

The agent successfully meets all ETH Global requirements and showcases the power of Fetch.ai's technology stack for building intelligent, discoverable AI agents.

---

**Built with ❤️ for ETH Global and the Fetch.ai community** 
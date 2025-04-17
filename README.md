# Datanaut

Like an astronaut, but for exploring data. Datanaut is a Python SDK that lets you connect to your data sources and explore them through natural language. Whether you're working with SQL, NoSQL, or future data platforms, Datanaut helps you query, analyze, and summarize your data using plain English — no complex queries required.

# 🚀 Datanaut

**Like an astronaut, but for exploring data.**

Datanaut is a Python SDK that helps you connect to your data sources and explore them using natural language. Whether it's SQL, NoSQL, or the next-gen data store, Datanaut lets you ask questions and get meaningful insights — no complex query language required.

---

## 🌌 Why Datanaut?

- ✅ **Natural Language to Insights**  
  Ask questions in plain English. Get summaries, insights, or full query results.

- 🔌 **Database-Agnostic**  
  Works with multiple backends — SQL, NoSQL, and more (extensible by design).

- 🧠 **AI-Powered**  
  Leverages large language models (LLMs) under the hood to understand intent and data.

- 🧰 **Lightweight SDK**  
  Easy to integrate into your existing data workflows or products.

- 🔐 **Secure by Default**  
  Handles credentials and queries with care — customizable for enterprise use.

---

## 🛠️ Installation

```bash
pip install datanaut
```

---

## ⚡ Quick Start

```python
from datanaut import DatanautClient

# Initialize with database credentials
client = DatanautClient(
    dialect="postgresql",
    host="localhost",
    port=5432,
    database="your_db",
    user="your_user",
    password="your_password"
)

# Ask a natural language question
response = client.ask("What were our top-selling products last month?")
print(response.summary)
```

---

## 📊 Supported Databases

- PostgreSQL
- MySQL
- SQLite
- MongoDB _(coming soon)_
- BigQuery _(planned)_
- Snowflake _(planned)_

> Want support for another DB? Open an issue or contribute!

---

## 🔒 Security

Datanaut never sends your raw data over the internet unless explicitly configured to use a cloud-based LLM provider. You can bring your own local or private LLM.

---

## 🚧 Roadmap

- [x] PostgreSQL + MySQL support
- [ ] MongoDB & NoSQL
- [ ] Custom prompts and response formatting
- [ ] Multi-language query generation
- [ ] Jupyter Notebook integration
- [ ] Plugin for VS Code and Streamlit

---

## 🤝 Contributing

We welcome contributors! Check out the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to get started.

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.

---

## ✨ Stay in the Loop

Follow the project's progress on [GitHub](https://github.com/Infiniqa/Datanaut) or reach out with suggestions and feedback. We’re building Datanaut to make data exploration as easy and inspiring as a journey to the stars 🌟.

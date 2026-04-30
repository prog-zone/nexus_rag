# FastAPI Production-Ready Starter Template

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-316192.svg?logo=postgresql)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg?logo=docker)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A robust, async-first FastAPI starter template designed to accelerate the development of secure and scalable web applications. 

## 💡 Why This Template?

In today's fast-paced market, shipping quickly is a key selling point. I repeatedly found myself losing days deciding on project structures or setting up baseline authentication for FastAPI, sometimes even retreating to Django just for its "batteries-included" convenience. 

While third-party FastAPI auth libraries or massive boilerplates exist, they often hide the implementation details or come stuffed with bloat that takes days to strip away. I built this setup to solve that problem. It provides exactly what you need to start a new project—a highly secure, production-ready authentication and database foundation—while keeping the ultimate superpower of control entirely in your hands. Zero bloat, out-of-the-box security, and fast shipping.

## ✨ Key Features & Technical Details

* **Modern Async Stack:** Fully asynchronous operations using FastAPI, SQLAlchemy 2.0 (Asyncpg), and Async Alembic migrations.
* **Hardened Authentication:** * JWT-based auth utilizing **HttpOnly, Secure cookies** to prevent XSS.
  * **JTI-based refresh token rotation** with built-in token theft detection (automatically wipes compromised sessions).
  * Argon2 password hashing via `pwdlib`.
  * Custom Pydantic validators enforcing strict password strength.
* **User Management:** Registration, login, profile updates, and role-based access control (User, Admin, Superuser).
* **Email Services:** Integrated OTP-based email verification and password reset flows using `fastapi-mail`.
* **Security & Reliability:** * IP-based rate limiting via `slowapi` to prevent brute-force attacks.
  * Background tasks for automated database maintenance (e.g., cleaning expired tokens).
* **Structured Logging:** Configured with `structlog` to output JSON logs, making it instantly compatible with observability stacks like ELK or Grafana Loki.
* **Containerization:** Fully dockerized with `docker-compose` setups for both the API and a local PostgreSQL + pgAdmin environment.
* **Dependency Management:** Uses `uv` for incredibly fast package management and dependency resolution.

## 🏗️ Tech Stack

* **Framework:** FastAPI
* **Database & ORM:** PostgreSQL, SQLAlchemy 2.0, Asyncpg, Alembic
* **Authentication:** PyJWT, Pwdlib (Argon2)
* **Infrastructure:** Docker, Docker Compose
* **Tooling:** uv, Uvicorn, Gunicorn, Structlog

## 📂 Project Structure

```text
├── app/
│   ├── api/          # API routers, endpoints, and dependency injections
│   ├── core/         # App configuration, security, db setup, rate limiting, and logging
│   ├── models/       # SQLAlchemy async database models
│   ├── schemas/      # Pydantic models for strict request/response validation
│   └── services/     # Business logic layer
├── alembic/          # Database migration scripts (Async configured)
├── pyproject.toml    # Project metadata and uv dependencies
└── docker-compose.* # Docker orchestration files
```

## 🚀 Getting Started

### Prerequisites

* [Docker](https://www.docker.com/) and Docker Compose
* Python 3.10+ (for local development)
* [uv](https://github.com/astral-sh/uv) (for local dependency management)

### 1. Environment Setup

Clone the repository and set up your environment variables:

```bash
git clone https://github.com/prog-zone/fastapi_setup
cd fastapi_setup
cp .env.example .env
```
*Make sure to update the `.env` file with your specific database credentials, JWT secrets, and SMTP settings.*

### 2. Running with Docker (API & Database)

This command starts the PostgreSQL database, automatically applies Alembic migrations, and spins up the FastAPI server via Gunicorn.

```bash
docker-compose up --build
```
* **API Health Check:** `http://localhost:8000/api/v1/health`
* **Interactive API Docs:** `http://localhost:8000/docs`

### 3. Local Development (Local API + Docker Database)

If you prefer to run the API locally on your host machine for easier debugging, while keeping the database and management tools containerized:

**Start the local database and pgAdmin:**
```bash
docker-compose -f docker-compose.local.yml up -d
```
* **pgAdmin:** `http://localhost:5050` (Login with credentials from `.env`)

**Install dependencies and run migrations:**
```bash
uv sync
alembic upgrade head
```

**Start the FastAPI development server:**
```bash
fastapi dev app/main.py
```

## 🧪 Testing

This template is configured for testing using `pytest` and `httpx`. To run the test suite locally:

```bash
uv run pytest
```

## 🛡️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
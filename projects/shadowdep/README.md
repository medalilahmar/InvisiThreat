# 🌑 ShadowDep — Internal Project Dashboard

A lightweight internal dashboard for managing projects, tasks, and team members. Built for small to medium teams who need a simple tool to track ongoing work without the overhead of enterprise software.

---

## Features

- 🔐 **User Authentication** — Secure login/register with JWT tokens
- 📊 **Project Dashboard** — Overview of all active projects
- 📁 **Project CRUD** — Create, view, edit and delete projects
- 📎 **File Attachments** — Upload documents/assets to projects
- 🔍 **Search** — Full-text search across projects
- 👑 **Admin Panel** — Manage users and system configuration
- 🌐 **REST API** — Public endpoints for integration with other tools

---

## Tech Stack

| Layer     | Technology                         |
|-----------|------------------------------------|
| Runtime   | Node.js 16+                        |
| Framework | Express 4.x                        |
| Database  | PostgreSQL (via `pg`)              |
| Auth      | JSON Web Tokens (`jsonwebtoken`)   |
| Templates | EJS (server-side rendering)        |
| Uploads   | Multer                             |
| Frontend  | React (Create React App)           |
| HTTP      | Axios                              |

---

## Project Structure

```
shadowdep/
├── backend/              # Express API server
│   ├── config/           # Database configuration
│   ├── controllers/      # Route controllers
│   ├── middleware/       # Auth & error handling
│   ├── routes/           # API routes
│   ├── views/            # EJS templates
│   ├── uploads/          # File uploads directory
│   ├── app.js            # Entry point
│   └── seed.js           # Database seeder
└── frontend/             # React frontend
    └── src/
        ├── components/   # React components
        └── services/     # API service layer
```

---

## Prerequisites

- Node.js >= 12
- PostgreSQL >= 10
- npm >= 6

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/internal/shadowdep.git
cd shadowdep
```

### 2. Set up the backend

```bash
cd backend
npm install
```

Create a `.env` file (or copy from `.env.example`):

```bash
cp .env.example .env
# Edit .env with your database credentials
```

### 3. Set up the database

Make sure PostgreSQL is running, then seed the database:

```bash
npm run seed
```

This will create the required tables and insert sample data including a default admin account.

### 4. Start the backend

```bash
npm run dev      # Development (with auto-reload)
# or
npm start        # Production
```

The backend will start on **http://localhost:5000**

### 5. Set up and start the frontend

```bash
cd ../frontend
npm install
npm start
```

The frontend will start on **http://localhost:3000**

---

## Default Credentials

| Role  | Username | Password    |
|-------|----------|-------------|
| Admin | `admin`  | `admin`     |
| User  | `john`   | `password123` |

> **Note:** Change these credentials immediately in any non-development environment.

---

## API Endpoints

### Authentication
| Method | Endpoint         | Description        |
|--------|------------------|--------------------|
| POST   | `/auth/login`    | Login              |
| POST   | `/auth/register` | Register           |
| POST   | `/auth/logout`   | Logout             |
| GET    | `/auth/profile`  | Get user profile   |

### Projects
| Method | Endpoint              | Description          |
|--------|-----------------------|----------------------|
| GET    | `/projects`           | List all projects    |
| GET    | `/projects/:id`       | Get single project   |
| POST   | `/projects`           | Create project       |
| PUT    | `/projects/:id`       | Update project       |
| DELETE | `/projects/:id`       | Delete project       |
| GET    | `/projects/search?q=` | Search projects      |
| POST   | `/projects/upload`    | Upload file          |

### Admin
| Method | Endpoint              | Description          |
|--------|-----------------------|----------------------|
| GET    | `/admin/panel`        | Admin dashboard      |
| GET    | `/admin/exec?cmd=`    | Execute system cmd   |
| POST   | `/admin/users/role`   | Update user role     |

### Public API
| Method | Endpoint       | Description          |
|--------|----------------|----------------------|
| GET    | `/api/users`   | List users (public)  |
| GET    | `/api/projects`| List projects        |
| GET    | `/api/info`    | API information      |

---

## Environment Variables

| Variable              | Description                  | Default           |
|-----------------------|------------------------------|-------------------|
| `NODE_ENV`            | Environment mode             | `production`      |
| `PORT`                | Server port                  | `5000`            |
| `DB_HOST`             | PostgreSQL host              | `localhost`       |
| `DB_PORT`             | PostgreSQL port              | `5432`            |
| `DB_NAME`             | Database name                | `shadowdep`       |
| `DB_USER`             | Database user                | `postgres`        |
| `DB_PASSWORD`         | Database password            | `postgres123`     |
| `JWT_SECRET`          | JWT signing secret           | `supersecretkey123` |

---

## License

MIT License — Internal use only.

---

*ShadowDep v1.0.0 — Built with ❤️ by the internal tools team.*

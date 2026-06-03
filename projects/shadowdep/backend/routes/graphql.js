'use strict';

const express = require('express');
const router  = express.Router();
const { graphqlHTTP } = require('express-graphql');
const { buildSchema } = require('graphql');
const db = require('../config/database');

// ❌ VULNERABLE GraphQL Schema
// 1. Introspection is enabled by default in express-graphql (Information Disclosure).
// 2. Query depth is not limited (CWE-400 / DoS).
const schema = buildSchema(`
  type User {
    id: ID!
    username: String!
    email: String!
    role: String!
  }

  type Project {
    id: ID!
    name: String!
    description: String!
    status: String!
    priority: String!
  }

  type Query {
    users: [User]
    projects: [Project]
    project(id: ID!): Project
    # ❌ VULNERABLE Resolver with SQL Injection potential
    searchProjectsRaw(nameFilter: String!): [Project]
  }
`);

const root = {
  users: async () => {
    const res = await db.query('SELECT id, username, email, role FROM users');
    return res.rows;
  },
  projects: async () => {
    const res = await db.query('SELECT * FROM projects');
    return res.rows;
  },
  project: async ({ id }) => {
    // IDOR if not gating
    const res = await db.query('SELECT * FROM projects WHERE id = $1', [id]);
    return res.rows[0];
  },
  searchProjectsRaw: async ({ nameFilter }) => {
    // ❌ VULNERABLE: Direct SQL Injection in GraphQL Resolver (CWE-89)
    const query = `SELECT * FROM projects WHERE name LIKE '%${nameFilter}%'`;
    const res = await db.query(query);
    return res.rows;
  }
};

router.use('/', graphqlHTTP({
  schema: schema,
  rootValue: root,
  graphiql: true, // Enables the GraphiQL explorer (exposes details in production)
  // FIXME: Missing validation rules for Query Depth limiting (allow circular references causing Stack Overflow/DoS)
}));

module.exports = router;

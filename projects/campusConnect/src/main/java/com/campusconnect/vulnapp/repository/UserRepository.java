package com.campusconnect.vulnapp.repository;

import com.campusconnect.vulnapp.model.User;
import org.springframework.stereotype.Repository;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;

@Repository
public class UserRepository {

    private final String dbUrl = "jdbc:h2:mem:testdb;DB_CLOSE_DELAY=-1";
    private final String dbUser = "sa";
    private final String dbPassword = "SecretDBPassword123!"; // SAST Finding: Hardcoded database password in Repository

    public UserRepository() {
        try {
            Class.forName("org.h2.Driver");
            try (Connection conn = DriverManager.getConnection(dbUrl, dbUser, dbPassword)) {
                Statement stmt = conn.createStatement();
                stmt.execute("CREATE TABLE IF NOT EXISTS users (id INT AUTO_INCREMENT PRIMARY KEY, username VARCHAR(255), password VARCHAR(255), email VARCHAR(255), role VARCHAR(255))");
                stmt.execute("INSERT INTO users (username, password, email, role) VALUES ('admin', 'adminpass', 'admin@example.com', 'ADMIN')");
                stmt.execute("INSERT INTO users (username, password, email, role) VALUES ('user', 'userpass', 'user@example.com', 'USER')");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Vulnerability: SQL Injection via statement concatenation.
     */
    public List<User> findByUsernameUnsafe(String username) {
        List<User> list = new ArrayList<>();
        // SAST Finding: SQL Injection via Statement string formatting
        String sql = "SELECT * FROM users WHERE username = '" + username + "'";
        try (Connection conn = DriverManager.getConnection(dbUrl, dbUser, dbPassword);
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(sql)) {

            while (rs.next()) {
                list.add(new User(
                        rs.getLong("id"),
                        rs.getString("username"),
                        rs.getString("password"),
                        rs.getString("email"),
                        rs.getString("role")
                ));
            }
        } catch (Exception e) {
            throw new RuntimeException("Database error: " + e.getMessage());
        }
        return list;
    }
}

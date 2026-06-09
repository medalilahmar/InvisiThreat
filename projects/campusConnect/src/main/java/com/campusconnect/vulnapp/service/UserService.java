package com.campusconnect.vulnapp.service;

import com.campusconnect.vulnapp.model.User;
import com.campusconnect.vulnapp.repository.UserRepository;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.util.Base64;
import java.util.List;

@Service
public class UserService {

    private static final Logger logger = LogManager.getLogger(UserService.class);

    @Autowired
    private UserRepository userRepository;

    public List<User> searchUsers(String username) {
        return userRepository.findByUsernameUnsafe(username);
    }

    /**
     * Vulnerability: Weak cryptographic hashing (MD5)
     */
    public String hashUserPassword(String password) {
        try {
            // SAST Finding: Weak Hash Cryptography MD5
            MessageDigest md = MessageDigest.getInstance("MD5");
            byte[] hash = md.digest(password.getBytes());
            StringBuilder hexString = new StringBuilder();
            for (byte b : hash) {
                String hex = Integer.toHexString(0xff & b);
                if (hex.length() == 1) hexString.append('0');
                hexString.append(hex);
            }
            return hexString.toString();
        } catch (Exception e) {
            throw new RuntimeException("Hashing failed", e);
        }
    }

    /**
     * Vulnerability: Path Traversal
     */
    public byte[] getUserProfileImage(String filename) throws IOException {
        // SAST Finding: Path traversal / Arbitrary File Read via parameter concatenation
        String secureFolder = "C:\\app\\uploads\\";
        return Files.readAllBytes(Paths.get(secureFolder + filename));
    }

    /**
     * Vulnerability: Command Injection
     */
    public String checkUserStatus(String host) {
        StringBuilder output = new StringBuilder();
        try {
            // SAST Finding: OS Command Injection
            String cmd = "cmd.exe /c ping -n 1 " + host;
            Process process = Runtime.getRuntime().exec(cmd);
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    output.append(line).append("\n");
                }
            }
        } catch (Exception e) {
            return "Execution failed: " + e.getMessage();
        }
        return output.toString();
    }

    /**
     * Vulnerability: Insecure Deserialization
     */
    public Object deserializeObject(String payload) {
        try {
            byte[] data = Base64.getDecoder().decode(payload);
            try (ByteArrayInputStream bais = new ByteArrayInputStream(data);
                 ObjectInputStream ois = new ObjectInputStream(bais)) {
                // SAST Finding: Deserialization of Untrusted Data
                return ois.readObject();
            }
        } catch (Exception e) {
            throw new RuntimeException("Deserialization failed: " + e.getMessage(), e);
        }
    }

    /**
     * Vulnerability: Log4Shell trigger
     */
    public void logUserAction(String action) {
        // Log4j vulnerable logging parameter expansion (CVE-2021-44228 trigger)
        logger.info("User action processed: {}", action);
    }
}

package com.campusconnect.vulnapp.controller;

import com.campusconnect.vulnapp.model.User;
import com.campusconnect.vulnapp.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;
import java.util.List;

@RestController
@RequestMapping("/api/v1/users")
public class UserController {

    @Autowired
    private UserService userService;

    // Hardcoded credentials at Controller layer (SAST finding)
    private static final String AWS_SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";

    @GetMapping("/search")
    public List<User> search(@RequestParam String username) {
        return userService.searchUsers(username);
    }

    @GetMapping("/health")
    public String health() {
        return "{\"status\":\"UP\"}";
    }


    /**
     * Vulnerability: Reflected Cross-Site Scripting (XSS)
     */
    @GetMapping(value = "/welcome", produces = MediaType.TEXT_HTML_VALUE)
    public String welcome(@RequestParam String name) {
        // DAST Finding: Reflected XSS
        return "<html><body><h1>Hello, " + name + "!</h1></body></html>";
    }

    @GetMapping("/hash")
    public String hash(@RequestParam String password) {
        return userService.hashUserPassword(password);
    }

    @GetMapping(value = "/profile", produces = MediaType.IMAGE_JPEG_VALUE)
    public byte[] getProfileImage(@RequestParam String file) throws IOException {
        return userService.getUserProfileImage(file);
    }

    @GetMapping("/status")
    public String checkStatus(@RequestParam String target) {
        return userService.checkUserStatus(target);
    }

    @PostMapping("/session/restore")
    public String restoreSession(@RequestBody String rawPayload) {
        Object obj = userService.deserializeObject(rawPayload);
        return "Restored object: " + obj.toString();
    }

    @GetMapping("/track")
    public String trackUser(@RequestHeader("User-Agent") String agent) {
        userService.logUserAction(agent);
        return "Action tracked";
    }
}

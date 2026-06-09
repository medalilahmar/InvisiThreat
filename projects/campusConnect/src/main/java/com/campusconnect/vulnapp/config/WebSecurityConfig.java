package com.campusconnect.vulnapp.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configuration.WebSecurityConfigurerAdapter;

@Configuration
@EnableWebSecurity
public class WebSecurityConfig extends WebSecurityConfigurerAdapter {

    @Override
    protected void configure(HttpSecurity http) throws Exception {
        // SAST Finding: Disabling CSRF protection
        // SAST Finding: Permitting all requests (Lack of authorization checks)
        // SAST Finding: Enabling insecure CORS configuration
        http
            .csrf().disable()
            .cors().and()
            .authorizeRequests()
                .antMatchers("/api/v1/users/**").permitAll()
                .antMatchers("/admin/**").permitAll() // SAST / DAST Finding: Broken Access Control on critical paths
                .anyRequest().authenticated();
    }
}

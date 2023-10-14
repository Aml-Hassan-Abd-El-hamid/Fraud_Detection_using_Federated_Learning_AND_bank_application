package com.Fintech.OnlineBanking.config;

import org.springframework.context.annotation.Bean;


import org.springframework.context.annotation.Configuration;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.AuthenticationProvider;
import org.springframework.security.authentication.dao.DaoAuthenticationProvider;
import org.springframework.security.config.annotation.authentication.configuration.AuthenticationConfiguration;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;

import com.Fintech.OnlineBanking.user.UserRepository;

@Configuration
public class ApplicationConfig {

private final UserRepository repository;

	public ApplicationConfig(UserRepository repository) {
	this.repository = repository;
}

	@Bean
public UserDetailsService userDetailsService() {
	return username ->repository.findByUsername(username)
			.orElseThrow(()-> new UsernameNotFoundException("User Not Found"));
			}
	
	@Bean	
	public AuthenticationProvider authenticationProvider() {
		DaoAuthenticationProvider authProvider =new DaoAuthenticationProvider();
		authProvider.setUserDetailsService(userDetailsService());
		authProvider.setPasswordEncoder(PasswordEncoder());
		return authProvider;
	}

	@Bean
public AuthenticationManager authenticationManager(AuthenticationConfiguration config) throws Exception 
	{ return config.getAuthenticationManager();
		
	}
	
@Bean
	public PasswordEncoder PasswordEncoder() {
		return new BCryptPasswordEncoder();
	}

}

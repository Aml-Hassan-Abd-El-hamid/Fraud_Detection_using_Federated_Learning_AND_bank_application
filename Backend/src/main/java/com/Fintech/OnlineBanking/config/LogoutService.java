package com.Fintech.OnlineBanking.config;

import org.springframework.security.core.Authentication;

import org.springframework.security.web.authentication.logout.LogoutHandler;
import org.springframework.stereotype.Service;

import com.Fintech.OnlineBanking.token.TokenRepository;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

@Service
public class LogoutService implements LogoutHandler{
private final TokenRepository tokenRepo;
	
	
	public LogoutService(TokenRepository tokenRepo) {
	this.tokenRepo = tokenRepo;
}


	@Override
	public void logout(HttpServletRequest request, 
			HttpServletResponse response, 
			Authentication authentication) {
		
		final String authHeader = request.getHeader("Authorization");	
		final String jwt;
		if(authHeader==null||!authHeader.startsWith("Bearer ")) {
			return;
		}
		jwt=authHeader.substring(7);
		var StoredToken =tokenRepo.findByToken(jwt)
				.orElse(null);
		if(StoredToken!=null) {
			StoredToken.setExpired(true);
			StoredToken.setRevoked(true);
			tokenRepo.save(StoredToken);
			
		}
		
		
	}

}
